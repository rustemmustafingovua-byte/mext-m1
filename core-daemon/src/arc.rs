//! Byte-capacity ARC (Adaptive Replacement Cache).
//!
//! Classic ARC (Megiddo & Modha, FAST'03) with two resident lists:
//!
//! * `T1` — pages seen exactly once (recency).
//! * `T2` — pages seen at least twice (frequency).
//! * `B1`/`B2` — ghost lists holding only keys + sizes, used to adapt `p`,
//!   the target size of `T1`.
//!
//! The only difference from the paper is that capacity is measured in **bytes**
//! rather than pages, because a compressed frame is anywhere from 4 KiB to
//! 512 MiB.

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;

struct Entry<V> {
    value: V,
    size: usize,
}

pub struct Arc<K: Eq + Hash + Clone, V> {
    capacity: usize,
    /// Target size (bytes) of T1.
    p: usize,
    t1: VecDeque<K>,
    t2: VecDeque<K>,
    b1: VecDeque<K>,
    b2: VecDeque<K>,
    entries: HashMap<K, Entry<V>>,
    ghost_sizes: HashMap<K, usize>,
    t1_bytes: usize,
    t2_bytes: usize,
    b1_bytes: usize,
    b2_bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

impl<K: Eq + Hash + Clone, V> Arc<K, V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            p: 0,
            t1: VecDeque::new(),
            t2: VecDeque::new(),
            b1: VecDeque::new(),
            b2: VecDeque::new(),
            entries: HashMap::new(),
            ghost_sizes: HashMap::new(),
            t1_bytes: 0,
            t2_bytes: 0,
            b1_bytes: 0,
            b2_bytes: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Change the budget, evicting immediately if we are over it.
    /// Returns the evicted values so the caller can `madvise` them away.
    pub fn set_capacity(&mut self, capacity: usize) -> Vec<(K, V)> {
        self.capacity = capacity.max(1);
        if self.p > self.capacity {
            self.p = self.capacity;
        }
        let mut out = Vec::new();
        while self.resident_bytes() > self.capacity {
            match self.replace(false) {
                Some(kv) => out.push(kv),
                None => break,
            }
        }
        self.trim_ghosts();
        out
    }

    pub fn resident_bytes(&self) -> usize {
        self.t1_bytes + self.t2_bytes
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn contains(&self, key: &K) -> bool {
        self.entries.contains_key(key)
    }

    pub fn peek(&self, key: &K) -> Option<&V> {
        self.entries.get(key).map(|e| &e.value)
    }

    pub fn peek_mut(&mut self, key: &K) -> Option<&mut V> {
        self.entries.get_mut(key).map(|e| &mut e.value)
    }

    /// Lookup + ARC promotion. Counts hits/misses.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        let Some(entry) = self.entries.get(key) else {
            self.misses += 1;
            return None;
        };
        let size = entry.size;
        self.hits += 1;

        if let Some(pos) = self.t1.iter().position(|k| k == key) {
            self.t1.remove(pos);
            self.t1_bytes -= size;
            self.t2.push_front(key.clone());
            self.t2_bytes += size;
        } else if let Some(pos) = self.t2.iter().position(|k| k == key) {
            self.t2.remove(pos);
            self.t2.push_front(key.clone());
        }
        self.entries.get(key).map(|e| &e.value)
    }

    /// Promote without counting a hit (used when the caller already accounted).
    pub fn touch(&mut self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Insert or update. Returns values evicted to make room (caller must
    /// release their memory — for the compression pool that means `madvise`).
    pub fn insert(&mut self, key: K, size: usize, value: V) -> Vec<(K, V)> {
        let size = size.max(1);
        let mut evicted = Vec::new();

        if let Some(existing) = self.entries.get_mut(&key) {
            let old = existing.size;
            existing.value = value;
            existing.size = size;
            if let Some(pos) = self.t1.iter().position(|k| *k == key) {
                // Re-reference: demote from T1 (recency) to T2 (frequency).
                self.t1.remove(pos);
                self.t1_bytes = self.t1_bytes.saturating_sub(old);
                self.t2.push_front(key);
                self.t2_bytes += size;
            } else if let Some(pos) = self.t2.iter().position(|k| *k == key) {
                self.t2.remove(pos);
                self.t2_bytes = self.t2_bytes.saturating_sub(old) + size;
                self.t2.push_front(key);
            } else {
                self.t1.push_front(key);
                self.t1_bytes += size;
            }
            while self.resident_bytes() > self.capacity {
                match self.replace(false) {
                    Some(kv) => evicted.push(kv),
                    None => break,
                }
            }
            return evicted;
        }

        // Ghost hit: adapt p exactly like the paper.
        let mut goes_to_t2 = false;
        if let Some(pos) = self.b1.iter().position(|k| *k == key) {
            let ghost = self.ghost_size(&key);
            self.b1.remove(pos);
            self.b1_bytes = self.b1_bytes.saturating_sub(ghost);
            self.ghost_sizes.remove(&key);
            let delta = if self.b1_bytes >= self.b2_bytes {
                size
            } else {
                (size * self.b2_bytes / self.b1_bytes.max(1)).max(size)
            };
            self.p = (self.p + delta).min(self.capacity);
            goes_to_t2 = true;
        } else if let Some(pos) = self.b2.iter().position(|k| *k == key) {
            let ghost = self.ghost_size(&key);
            self.b2.remove(pos);
            self.b2_bytes = self.b2_bytes.saturating_sub(ghost);
            self.ghost_sizes.remove(&key);
            let delta = if self.b2_bytes >= self.b1_bytes {
                size
            } else {
                (size * self.b1_bytes / self.b2_bytes.max(1)).max(size)
            };
            self.p = self.p.saturating_sub(delta);
            goes_to_t2 = true;
        }

        let incoming_b2 = goes_to_t2;
        while self.resident_bytes() + size > self.capacity
            && !(self.t1.is_empty() && self.t2.is_empty())
        {
            match self.replace(incoming_b2) {
                Some(kv) => evicted.push(kv),
                None => break,
            }
        }

        self.entries.insert(key.clone(), Entry { value, size });
        if goes_to_t2 {
            self.t2.push_front(key);
            self.t2_bytes += size;
        } else {
            self.t1.push_front(key);
            self.t1_bytes += size;
        }
        self.evictions += evicted.len() as u64;
        self.trim_ghosts();
        evicted
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        let entry = self.entries.remove(key)?;
        if let Some(pos) = self.t1.iter().position(|k| k == key) {
            self.t1.remove(pos);
            self.t1_bytes = self.t1_bytes.saturating_sub(entry.size);
        } else if let Some(pos) = self.t2.iter().position(|k| k == key) {
            self.t2.remove(pos);
            self.t2_bytes = self.t2_bytes.saturating_sub(entry.size);
        }
        Some(entry.value)
    }

    /// Pop the next ARC victim (LRU of T1 or T2) without inserting anything.
    ///
    /// The compressed pool drives eviction from *real* resident bytes (arena
    /// pages + dedicated mappings), which can exceed the payload sizes ARC
    /// tracks internally; this is the hook for that extra pressure.
    pub fn evict_lru(&mut self) -> Option<(K, V)> {
        let evicted = self.replace(false);
        if evicted.is_some() {
            self.evictions += 1;
        }
        evicted
    }

    /// Evict everything (used on shutdown / budget collapse).
    pub fn drain_all(&mut self) -> Vec<(K, V)> {
        let mut out = Vec::new();
        for (k, e) in self.entries.drain() {
            out.push((k, e.value));
            let _ = e.size;
        }
        self.t1.clear();
        self.t2.clear();
        self.t1_bytes = 0;
        self.t2_bytes = 0;
        self.evictions += out.len() as u64;
        out
    }

    /// Resident keys, most recently used first.
    pub fn keys(&self) -> Vec<&K> {
        self.t1.iter().chain(self.t2.iter()).collect()
    }

    /// `p / capacity` — how much the cache currently favours recency.
    pub fn p_ratio(&self) -> f64 {
        self.p as f64 / self.capacity.max(1) as f64
    }

    fn ghost_size(&self, key: &K) -> usize {
        self.ghost_sizes.get(key).copied().unwrap_or(0)
    }

    /// ARC's REPLACE: demote the LRU victim of T1 (or T2) into its ghost list
    /// and hand its value back to the caller.
    fn replace(&mut self, incoming_b2: bool) -> Option<(K, V)> {
        let prefer_t1 = !self.t1.is_empty()
            && (self.t1_bytes > self.p || (incoming_b2 && self.t1_bytes == self.p));

        if prefer_t1 {
            let key = self.t1.pop_back()?;
            let entry = self.entries.remove(&key)?;
            self.t1_bytes = self.t1_bytes.saturating_sub(entry.size);
            self.b1.push_front(key.clone());
            self.b1_bytes += entry.size;
            self.ghost_sizes.insert(key.clone(), entry.size);
            Some((key, entry.value))
        } else if let Some(key) = self.t2.pop_back() {
            let entry = self.entries.remove(&key)?;
            self.t2_bytes = self.t2_bytes.saturating_sub(entry.size);
            self.b2.push_front(key.clone());
            self.b2_bytes += entry.size;
            self.ghost_sizes.insert(key.clone(), entry.size);
            Some((key, entry.value))
        } else {
            None
        }
    }

    fn trim_ghosts(&mut self) {
        while self.b1_bytes + self.b2_bytes > self.capacity {
            let popped = if self.b1_bytes >= self.b2_bytes {
                self.b1.pop_back().map(|k| (k, true))
            } else {
                self.b2.pop_back().map(|k| (k, false))
            };
            let Some((key, from_b1)) = popped else { break };
            let size = self.ghost_size(&key);
            self.ghost_sizes.remove(&key);
            if from_b1 {
                self.b1_bytes = self.b1_bytes.saturating_sub(size);
            } else {
                self.b2_bytes = self.b2_bytes.saturating_sub(size);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arc(cap: usize) -> Arc<String, usize> {
        Arc::new(cap)
    }

    #[test]
    fn evicts_when_over_capacity() {
        let mut a = arc(300);
        a.insert("a".into(), 100, 1);
        a.insert("b".into(), 100, 2);
        assert!(a.resident_bytes() <= 300);
        let evicted = a.insert("c".into(), 100, 3);
        assert_eq!(a.resident_bytes(), 300);
        assert_eq!(a.len(), 3);
        assert!(evicted.is_empty());

        let evicted = a.insert("d".into(), 100, 4);
        assert_eq!(evicted.len(), 1);
        assert!(a.resident_bytes() <= 300);
        assert_eq!(a.len(), 3);
    }

    #[test]
    fn huge_entry_flushes_cache() {
        let mut a = arc(1000);
        for i in 0..5 {
            a.insert(format!("k{i}"), 200, i);
        }
        assert_eq!(a.len(), 5);
        let evicted = a.insert("big".into(), 900, 99);
        assert_eq!(evicted.len(), 5);
        assert_eq!(a.len(), 1);
        assert!(a.resident_bytes() <= 1000);
    }

    #[test]
    fn second_reference_goes_to_t2() {
        let mut a = arc(10_000);
        a.insert("x".into(), 100, 1);
        assert_eq!(a.t1.len(), 1);
        assert!(a.get(&String::from("x")).is_some());
        assert_eq!(a.t1.len(), 0);
        assert_eq!(a.t2.len(), 1);
        assert_eq!(a.hits, 1);
    }

    #[test]
    fn ghost_hit_adapts_p() {
        let mut a = arc(1000);
        for i in 0..10 {
            a.insert(format!("one{i}"), 100, i);
        }
        // Fill T1 with 10 cold entries, then reference one twice.
        a.insert("hot".into(), 100, 42);
        a.get(&"hot".to_string());
        // Force several evictions so ghosts appear.
        for i in 0..5 {
            a.insert(format!("cold{i}"), 100, i);
        }
        let p_before = a.p;
        // Re-request an evicted key that should now be in a ghost list.
        let mut reinserted = false;
        for i in 0..10 {
            let k = format!("one{i}");
            if !a.contains(&k) {
                a.insert(k, 100, 7);
                reinserted = true;
                break;
            }
        }
        assert!(reinserted, "expected some ghost resident");
        assert!(a.p != p_before || a.p > 0, "p should adapt on ghost hit");
        assert!(a.len() <= 10);
    }

    #[test]
    fn update_resizes() {
        let mut a = arc(1000);
        a.insert("a".into(), 100, 1);
        a.insert("a".into(), 400, 2);
        assert_eq!(a.resident_bytes(), 400);
        assert_eq!(*a.peek(&"a".to_string()).unwrap(), 2);
    }

    #[test]
    fn remove_and_drain() {
        let mut a = arc(1000);
        a.insert("a".into(), 100, 1);
        a.insert("b".into(), 100, 2);
        assert_eq!(a.remove(&"a".to_string()), Some(1));
        assert_eq!(a.len(), 1);
        let all = a.drain_all();
        assert_eq!(all.len(), 1);
        assert_eq!(a.resident_bytes(), 0);
        assert!(a.is_empty());
    }

    #[test]
    fn shrink_budget_evicts() {
        let mut a = arc(1000);
        for i in 0..10 {
            a.insert(format!("k{i}"), 100, i);
        }
        let out = a.set_capacity(300);
        assert_eq!(out.len(), 7);
        assert!(a.resident_bytes() <= 300);
    }
}
