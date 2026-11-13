//! `CompressedMemoryPool` — a hard-capped, LZ4-compressed RAM tier.
//!
//! Design constraints that come from the task:
//!
//! 1. **Hard RAM budget.** `--ram-budget 2GB` must never be exceeded, so the
//!    budget is enforced in *real resident bytes* (arena pages + dedicated
//!    mappings), not in "file bytes".
//! 2. **ARC eviction.** Replacement policy is Adaptive Replacement Cache, so
//!    both recency (recently touched) and frequency (hot files) are respected.
//! 3. **Real page release.** Dedicated frames live in their own anonymous
//!    mapping; small frames are packed into a refcounted [`SmallArena`]. On
//!    eviction we call `madvise(MADV_DONTNEED)` on the pages that became dead,
//!    so the memory is returned to macOS immediately.
//! 4. **Random access.** Files are split into `chunk_size` blocks, each block
//!    compressed independently, so reading 4 KiB does not decompress 40 MiB.
//!
//! Small files are the common case in a code workspace; giving each one a
//! 16 KiB page would triple the RAM bill, so payloads up to half a page share
//! packed pages instead.

use crate::arc::Arc;
use crate::arena::{ArenaHandle, SmallArena};
use crate::memmap::PageBuf;
use anyhow::{Context, Result};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WarmOutcome {
    /// Already resident and up to date.
    Hit,
    /// Read from SSD, compressed and inserted.
    Warmed,
    /// Deliberately ignored (too large / not a file / empty).
    Skipped,
}

#[derive(Debug, Clone, Serialize)]
pub struct WarmResult {
    pub outcome: WarmOutcome,
    pub logical_bytes: u64,
    pub compressed_bytes: u64,
    pub elapsed_ms: f64,
    pub evicted: usize,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct PoolStats {
    pub budget_bytes: u64,
    /// Real RSS impact: packed arena pages + dedicated page-rounded frames.
    pub resident_bytes: u64,
    /// Bytes resident frames occupy when decompressed ("virtual volume").
    pub logical_bytes: u64,
    /// Sum of the LZ4 payloads (excluding page-rounding waste).
    pub compressed_bytes: u64,
    /// Page-rounding / packing slack: resident_bytes - compressed_bytes.
    pub overhead_bytes: u64,
    pub arena_bytes: u64,
    pub arena_pages: u64,
    pub arena_frames: u64,
    pub dedicated_bytes: u64,
    pub dedicated_frames: usize,
    pub files: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub released_bytes: u64,
    pub ssd_read_bytes: u64,
    pub decompressions: u64,
    pub warmed: u64,
    pub warm_ms_last: f64,
    pub ratio: f64,
    /// ARC adaptation: share of the budget currently given to recency (T1).
    pub arc_p_ratio: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PoolEntry {
    pub path: String,
    pub logical_bytes: u64,
    pub compressed_bytes: u64,
    pub ratio: f64,
    /// `packed` (shared arena page) or `dedicated` (own mapping).
    pub storage: &'static str,
}

struct Chunk {
    comp_len: usize,
    raw_len: usize,
}

enum FrameData {
    /// Frame has its own page-rounded mapping.
    Held(PageBuf),
    /// Frame shares a refcounted arena page with neighbours.
    Arena(ArenaHandle),
}

struct Frame {
    data: FrameData,
    chunks: Vec<Chunk>,
    logical_len: usize,
    mtime_ns: i64,
    size: u64,
    hits: u64,
}

impl Frame {
    fn compressed_len(&self) -> usize {
        self.chunks.iter().map(|c| c.comp_len).sum()
    }

    /// What ARC accounts for this frame. Dedicated frames are charged their
    /// mapped size (rounding included); packed frames their payload, because
    /// the page is shared. The hard budget uses [`Pool::real_resident`].
    fn charged(&self) -> usize {
        match &self.data {
            FrameData::Held(buf) => buf.mapped_len(),
            FrameData::Arena(_) => self.compressed_len().max(1),
        }
    }

    fn is_dedicated(&self) -> bool {
        matches!(self.data, FrameData::Held(_))
    }

    fn view<'a>(&'a self, arena: &'a SmallArena) -> &'a [u8] {
        match &self.data {
            FrameData::Held(buf) => buf.as_slice(),
            FrameData::Arena(handle) => arena.view(handle),
        }
    }

    /// Return the pages to XNU. Counts only bytes actually released.
    fn release(&mut self, arena: &mut SmallArena) -> u64 {
        match &mut self.data {
            FrameData::Held(buf) => {
                let bytes = buf.mapped_len() as u64;
                let _ = buf.dont_need();
                bytes
            }
            FrameData::Arena(handle) => arena.free(*handle),
        }
    }

    fn storage(&self) -> &'static str {
        match &self.data {
            FrameData::Held(_) => "dedicated",
            FrameData::Arena(_) => "packed",
        }
    }
}

pub struct CompressedMemoryPool {
    arc: Arc<PathBuf, Frame>,
    arena: SmallArena,
    budget: u64,
    chunk_size: usize,
    max_file_bytes: u64,
    logical_resident: u64,
    compressed_resident: u64,
    dedicated_bytes: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
    released_bytes: u64,
    ssd_read_bytes: u64,
    decompressions: u64,
    warmed: u64,
    warm_ms_last: f64,
}

impl CompressedMemoryPool {
    pub fn new(budget_bytes: u64, chunk_size: usize, max_file_bytes: u64) -> Self {
        let arena = SmallArena::new();
        let budget = budget_bytes.max(arena.page_bytes());
        Self {
            // ARC capacity is a lower bound of the real cost; the authoritative
            // limit is enforced in `enforce_budget`.
            arc: Arc::new(budget as usize),
            arena,
            budget,
            chunk_size: chunk_size.max(64 * 1024),
            max_file_bytes,
            logical_resident: 0,
            compressed_resident: 0,
            dedicated_bytes: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
            released_bytes: 0,
            ssd_read_bytes: 0,
            decompressions: 0,
            warmed: 0,
            warm_ms_last: 0.0,
        }
    }

    pub fn budget(&self) -> u64 {
        self.budget
    }

    /// Real RSS impact of everything resident right now.
    pub fn real_resident(&self) -> u64 {
        self.dedicated_bytes + self.arena.live_bytes()
    }

    /// Change the hard budget. Evicts (with `madvise`) down to the new limit.
    pub fn set_budget(&mut self, budget: u64) {
        self.budget = budget.max(self.arena.page_bytes());
        let evicted = self.arc.set_capacity(self.budget as usize);
        self.account_evictions(evicted);
        self.enforce_budget();
    }

    pub fn contains(&self, path: &Path) -> bool {
        self.arc.contains(&path.to_path_buf())
    }

    pub fn files(&self) -> usize {
        self.arc.len()
    }

    /// Warm a file into the pool. This is the main prefetch entry point.
    pub fn warm(&mut self, path: &Path) -> Result<WarmResult> {
        let started = Instant::now();
        let meta = match std::fs::metadata(path) {
            Ok(m) => m,
            Err(_) => {
                return Ok(WarmResult {
                    outcome: WarmOutcome::Skipped,
                    logical_bytes: 0,
                    compressed_bytes: 0,
                    elapsed_ms: 0.0,
                    evicted: 0,
                })
            }
        };
        if !meta.is_file() || meta.len() == 0 || meta.len() > self.max_file_bytes {
            return Ok(WarmResult {
                outcome: WarmOutcome::Skipped,
                logical_bytes: 0,
                compressed_bytes: 0,
                elapsed_ms: 0.0,
                evicted: 0,
            });
        }

        let key = canonical(path);
        let (size, mtime_ns) = stamp(&meta);

        let cached = self.arc.peek(&key).map(|frame| {
            (
                frame.size,
                frame.mtime_ns,
                frame.logical_len,
                frame.compressed_len(),
            )
        });
        if let Some((cached_size, cached_mtime, cached_logical, cached_compressed)) = cached {
            if cached_size == size && cached_mtime == mtime_ns {
                let elapsed = started.elapsed().as_secs_f64() * 1000.0;
                let _ = self.arc.get(&key);
                if let Some(f) = self.arc.peek_mut(&key) {
                    f.hits += 1;
                }
                self.hits += 1;
                return Ok(WarmResult {
                    outcome: WarmOutcome::Hit,
                    logical_bytes: cached_logical as u64,
                    compressed_bytes: cached_compressed as u64,
                    elapsed_ms: elapsed,
                    evicted: 0,
                });
            }
            // Stale: drop the old frame properly (madvise included).
            if let Some(old) = self.arc.remove(&key) {
                self.account_evictions(vec![(key.clone(), old)]);
            }
        }

        let data = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        self.ssd_read_bytes += data.len() as u64;

        // Compress chunk by chunk.
        let mut compressed: Vec<Vec<u8>> = Vec::with_capacity(data.len() / self.chunk_size + 1);
        let mut raw_lens: Vec<usize> = Vec::new();
        for chunk in data.chunks(self.chunk_size) {
            compressed.push(lz4_flex::block::compress(chunk));
            raw_lens.push(chunk.len());
        }
        let total_compressed: usize = compressed.iter().map(|c| c.len()).sum();

        let mut flat: Vec<u8> = Vec::with_capacity(total_compressed);
        for c in &compressed {
            flat.extend_from_slice(c);
        }

        let packed = self
            .arena
            .alloc_copy(&flat)
            .context("allocating packed arena frame")?;
        let storage = match packed {
            Some(handle) => FrameData::Arena(handle),
            None => {
                let mut buf = PageBuf::zeroed(total_compressed.max(1))
                    .context("allocating compressed frame mapping")?;
                buf.write(&flat);
                buf.will_need().ok();
                FrameData::Held(buf)
            }
        };
        drop(flat);

        let mut chunks = Vec::with_capacity(compressed.len());
        for (c, raw) in compressed.iter().zip(raw_lens.iter()) {
            chunks.push(Chunk {
                comp_len: c.len(),
                raw_len: *raw,
            });
        }
        drop(compressed);

        let frame = Frame {
            data: storage,
            chunks,
            logical_len: data.len(),
            mtime_ns,
            size,
            hits: 0,
        };
        let charged = frame.charged();
        let dedicated = frame.is_dedicated();
        let logical = frame.logical_len as u64;
        let inserted = self.arc.insert(key, charged, frame);
        self.logical_resident += logical;
        self.compressed_resident += total_compressed as u64;
        if dedicated {
            self.dedicated_bytes += charged as u64;
        }
        self.warmed += 1;
        let evicted_count = inserted.len();
        self.account_evictions(inserted);
        self.enforce_budget();

        self.warm_ms_last = started.elapsed().as_secs_f64() * 1000.0;
        Ok(WarmResult {
            outcome: WarmOutcome::Warmed,
            logical_bytes: logical,
            compressed_bytes: total_compressed as u64,
            elapsed_ms: self.warm_ms_last,
            evicted: evicted_count,
        })
    }

    /// Read `len` bytes at `offset`. Decompresses only the touched chunks.
    /// A miss transparently warms the file first (that is the SSD fallback).
    pub fn read(&mut self, path: &Path, offset: u64, len: usize) -> Result<Vec<u8>> {
        let key = canonical(path);
        if !self.arc.contains(&key) {
            self.misses += 1;
            self.warm(path)?;
        }

        let start = offset as usize;
        let mut out = Vec::new();
        let mut decompressed = 0u64;
        {
            // Disjoint field borrows: the frame comes from the ARC, the payload
            // may live in the arena. Neither is mutated in this block.
            let arc = &mut self.arc;
            let arena = &self.arena;
            let frame = arc
                .get(&key)
                .ok_or_else(|| anyhow::anyhow!("cannot warm {}", path.display()))?;
            if frame.logical_len == 0 || start >= frame.logical_len {
                return Ok(Vec::new());
            }
            let end = (start + len).min(frame.logical_len);
            out.reserve(end - start);
            let data = frame.view(arena);
            let mut logical_pos = 0usize;
            let mut comp_pos = 0usize;
            for chunk in &frame.chunks {
                let chunk_start = logical_pos;
                let chunk_end = logical_pos + chunk.raw_len;
                logical_pos = chunk_end;
                let src = &data[comp_pos..comp_pos + chunk.comp_len];
                comp_pos += chunk.comp_len;

                if chunk_end <= start || chunk_start >= end {
                    continue;
                }
                let mut decoded = vec![0u8; chunk.raw_len];
                let written = lz4_flex::block::decompress_into(src, &mut decoded)
                    .map_err(|e| anyhow::anyhow!("lz4 decompress: {e}"))?;
                debug_assert_eq!(written, chunk.raw_len);
                let lo = start.saturating_sub(chunk_start);
                let hi = (end - chunk_start).min(chunk.raw_len);
                out.extend_from_slice(&decoded[lo..hi]);
                decompressed += 1;
            }
        }
        self.decompressions += decompressed;
        Ok(out)
    }

    /// Whole file via the pool (used by the benchmark and by the AI bridge).
    pub fn read_all(&mut self, path: &Path) -> Result<Vec<u8>> {
        let size = std::fs::metadata(path)
            .map(|m| m.len() as usize)
            .unwrap_or(0);
        self.read(path, 0, size.max(1))
    }

    /// Mark a file as used without decompressing (ARC promotion only).
    pub fn touch(&mut self, path: &Path) -> bool {
        let key = canonical(path);
        let hit = self.arc.get(&key).is_some();
        if hit {
            self.hits += 1;
            if let Some(f) = self.arc.peek_mut(&key) {
                f.hits += 1;
            }
        }
        hit
    }

    /// Evict everything, releasing all pages via madvise/munmap.
    pub fn release_all(&mut self) -> u64 {
        let before = self.released_bytes;
        let all = self.arc.drain_all();
        self.account_evictions(all);
        self.dedicated_bytes = 0;
        self.logical_resident = 0;
        self.compressed_resident = 0;
        self.released_bytes - before
    }

    /// ARC-driven evictions (policy decision already made).
    fn account_evictions(&mut self, evicted: Vec<(PathBuf, Frame)>) {
        for (_key, mut frame) in evicted {
            self.logical_resident = self
                .logical_resident
                .saturating_sub(frame.logical_len as u64);
            self.compressed_resident = self
                .compressed_resident
                .saturating_sub(frame.compressed_len() as u64);
            if frame.is_dedicated() {
                self.dedicated_bytes = self.dedicated_bytes.saturating_sub(frame.charged() as u64);
            }
            let released = frame.release(&mut self.arena);
            self.released_bytes += released;
            self.evictions += 1;
        }
    }

    /// Enforce the hard budget from *real* resident bytes. Packed pages may hold
    /// several frames, so a single eviction might not free anything yet; we keep
    /// going until either we fit or the ARC is empty.
    fn enforce_budget(&mut self) {
        while self.real_resident() > self.budget {
            let Some(victim) = self.arc.evict_lru() else {
                break;
            };
            self.account_evictions(vec![victim]);
        }
    }

    pub fn stats(&self) -> PoolStats {
        let logical = self.logical_resident;
        let resident = self.real_resident();
        let compressed = self.compressed_resident.min(resident);
        PoolStats {
            budget_bytes: self.budget,
            resident_bytes: resident,
            logical_bytes: logical,
            compressed_bytes: compressed,
            overhead_bytes: resident.saturating_sub(compressed),
            arena_bytes: self.arena.live_bytes(),
            arena_pages: self.arena.live_pages(),
            arena_frames: self.arena.live_frames(),
            dedicated_bytes: self.dedicated_bytes,
            dedicated_frames: self
                .arc
                .keys()
                .len()
                .saturating_sub(self.arena.live_frames() as usize),
            files: self.arc.len(),
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            released_bytes: self.released_bytes,
            ssd_read_bytes: self.ssd_read_bytes,
            decompressions: self.decompressions,
            warmed: self.warmed,
            warm_ms_last: self.warm_ms_last,
            // LZ4 ratio of what is actually resident (page rounding excluded,
            // it is reported separately as `overhead_bytes`).
            ratio: if compressed == 0 {
                0.0
            } else {
                logical as f64 / compressed as f64
            },
            arc_p_ratio: self.arc.p_ratio(),
        }
    }

    /// Resident frames, largest first (dashboard "pool contents").
    pub fn entries(&self, limit: usize) -> Vec<PoolEntry> {
        let mut out: Vec<PoolEntry> = self
            .arc
            .keys()
            .into_iter()
            .filter_map(|key| {
                let frame = self.arc.peek(key)?;
                Some(PoolEntry {
                    path: key.display().to_string(),
                    logical_bytes: frame.logical_len as u64,
                    compressed_bytes: frame.compressed_len() as u64,
                    ratio: frame.logical_len as f64 / frame.compressed_len().max(1) as f64,
                    storage: frame.storage(),
                })
            })
            .collect();
        out.sort_by(|a, b| b.logical_bytes.cmp(&a.logical_bytes));
        out.truncate(limit);
        out
    }

    /// Bytes handed back to the OS so far (madvise + munmap).
    pub fn released_bytes(&self) -> u64 {
        self.released_bytes
    }
}

fn canonical(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
    }
}

fn stamp(meta: &std::fs::Metadata) -> (u64, i64) {
    use std::os::unix::fs::MetadataExt;
    let mtime_ns = meta.mtime() as i64 * 1_000_000_000 + meta.mtime_nsec() as i64;
    (meta.len(), mtime_ns)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let p = std::env::temp_dir().join(format!(
                "mext-pool-{tag}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .subsec_nanos()
            ));
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
        fn file(&self, name: &str, bytes: usize) -> PathBuf {
            let p = self.0.join(name);
            // Pseudo-random, incompressible payload so eviction tests are
            // deterministic (LZ4 would otherwise shrink patterned data 100×).
            let mut state = 0x1234_5678_9abc_def0u64 ^ bytes as u64;
            let mut data = Vec::with_capacity(bytes);
            for _ in 0..bytes {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                data.push((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) as u8);
            }
            std::fs::write(&p, data).unwrap();
            p
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).ok();
        }
    }

    #[test]
    fn warm_then_hit() {
        let dir = TempDir::new("hit");
        let f = dir.file("a.txt", 300_000);
        let mut pool = CompressedMemoryPool::new(64 << 20, 64 * 1024, 1 << 30);
        let first = pool.warm(&f).unwrap();
        assert_eq!(first.outcome, WarmOutcome::Warmed);
        assert_eq!(first.logical_bytes, 300_000);
        let second = pool.warm(&f).unwrap();
        assert_eq!(second.outcome, WarmOutcome::Hit);
        assert_eq!(pool.stats().hits, 1);
        assert_eq!(pool.stats().warmed, 1);
        assert!(pool.stats().resident_bytes >= 1);
    }

    #[test]
    fn random_read_roundtrip() {
        let dir = TempDir::new("read");
        let f = dir.file("data.bin", 500_000);
        let mut pool = CompressedMemoryPool::new(64 << 20, 64 * 1024, 1 << 30);
        let whole = pool.read_all(&f).unwrap();
        let original = std::fs::read(&f).unwrap();
        assert_eq!(whole, original);

        let slice = pool.read(&f, 123_456, 4096).unwrap();
        assert_eq!(slice, original[123_456..123_456 + 4096]);
        // The very first read_all was a cold miss (it had to warm the file).
        assert_eq!(pool.stats().misses, 1);
        assert!(pool.stats().decompressions >= 2);
    }

    #[test]
    fn small_files_share_arena_pages() {
        let dir = TempDir::new("arena");
        let mut files = Vec::new();
        for i in 0..16 {
            files.push(dir.file(&format!("small{i}.rs"), 2_000)); // ~2 KiB payload
        }
        let mut pool = CompressedMemoryPool::new(4 << 20, 64 * 1024, 1 << 30);
        for f in &files {
            pool.warm(f).unwrap();
        }
        let stats = pool.stats();
        let page = SmallArena::new().page_bytes();
        assert_eq!(stats.arena_frames, files.len() as u64);
        assert!(stats.arena_pages <= 16, "pages {}", stats.arena_pages);
        // Packed storage must beat one page per file.
        assert!(
            stats.resident_bytes < files.len() as u64 * page,
            "resident {} vs 16 pages",
            stats.resident_bytes
        );
        // Data integrity through the arena path.
        for f in &files {
            assert_eq!(pool.read_all(f).unwrap(), std::fs::read(f).unwrap());
        }
        assert!(stats.arena_bytes > 0);
    }

    #[test]
    fn arena_pages_are_reclaimed_when_empty() {
        let dir = TempDir::new("arena-release");
        let mut files = Vec::new();
        for i in 0..20 {
            files.push(dir.file(&format!("s{i}.rs"), 1_500));
        }
        let mut pool = CompressedMemoryPool::new(4 << 20, 64 * 1024, 1 << 30);
        for f in &files {
            pool.warm(f).unwrap();
        }
        let before = pool.stats();
        assert!(before.arena_pages >= 1);

        let page = SmallArena::new().page_bytes();
        pool.set_budget(page); // one page: half the packed frames must go
        let after = pool.stats();
        assert!(after.arena_pages < before.arena_pages, "{:?}", after);
        assert!(after.released_bytes > 0);
        assert!(after.resident_bytes <= page);
    }

    #[test]
    fn budget_is_hard_and_releases_pages() {
        let dir = TempDir::new("budget");
        let mut files = Vec::new();
        for i in 0..8 {
            files.push(dir.file(&format!("f{i}.bin"), 200_000));
        }
        let mut pool = CompressedMemoryPool::new(400_000, 64 * 1024, 1 << 30);
        for f in &files {
            pool.warm(f).unwrap();
        }
        let stats = pool.stats();
        assert!(stats.resident_bytes <= pool.budget(), "budget exceeded");
        assert!(stats.evictions > 0);
        assert!(stats.released_bytes > 0);
        assert!(stats.files >= 1);

        let released_before = stats.released_bytes;
        pool.set_budget(100_000);
        assert!(pool.stats().resident_bytes <= 200_000);
        assert!(pool.stats().released_bytes > released_before);
    }

    #[test]
    fn mixed_sizes_use_both_storages() {
        let dir = TempDir::new("mixed");
        let small = dir.file("tiny.rs", 900);
        let big = dir.file("big.bin", 400_000);
        let mut pool = CompressedMemoryPool::new(8 << 20, 64 * 1024, 1 << 30);
        pool.warm(&small).unwrap();
        pool.warm(&big).unwrap();
        let stats = pool.stats();
        assert_eq!(stats.arena_frames, 1);
        assert_eq!(stats.dedicated_frames, 1);
        assert!(stats.dedicated_bytes > 0);
        assert_eq!(
            pool.read_all(&small).unwrap(),
            std::fs::read(&small).unwrap()
        );
        assert_eq!(pool.read_all(&big).unwrap(), std::fs::read(&big).unwrap());
        let entries = pool.entries(10);
        assert_eq!(entries.len(), 2);
        let storages: Vec<&str> = entries.iter().map(|e| e.storage).collect();
        assert!(storages.contains(&"packed"));
        assert!(storages.contains(&"dedicated"));
    }

    #[test]
    fn skips_oversized() {
        let dir = TempDir::new("skip");
        let f = dir.file("big.bin", 2000);
        let mut pool = CompressedMemoryPool::new(1 << 20, 64 * 1024, 1000);
        assert_eq!(pool.warm(&f).unwrap().outcome, WarmOutcome::Skipped);
        assert_eq!(pool.files(), 0);
    }

    #[test]
    fn compression_ratio_positive() {
        let dir = TempDir::new("ratio");
        // Highly compressible data, larger than the arena threshold.
        let p = dir.0.join("zero.bin");
        std::fs::write(&p, vec![0u8; 2_000_000]).unwrap();
        let mut pool = CompressedMemoryPool::new(64 << 20, 256 * 1024, 1 << 30);
        let r = pool.warm(&p).unwrap();
        assert_eq!(r.logical_bytes, 2_000_000);
        assert!(r.compressed_bytes < r.logical_bytes / 10);
        assert!(pool.stats().ratio > 5.0);
        assert!(pool.stats().overhead_bytes < pool.stats().resident_bytes);
    }

    #[test]
    fn missing_file_is_skipped() {
        let mut pool = CompressedMemoryPool::new(1 << 20, 64 * 1024, 1 << 30);
        assert_eq!(
            pool.warm(Path::new("/definitely/not/here.rs"))
                .unwrap()
                .outcome,
            WarmOutcome::Skipped
        );
    }

    #[test]
    fn release_all_zeroes_resident() {
        let dir = TempDir::new("release");
        let small = dir.file("s.rs", 1000);
        let big = dir.file("x.bin", 100_000);
        let mut pool = CompressedMemoryPool::new(10 << 20, 64 * 1024, 1 << 30);
        pool.warm(&small).unwrap();
        pool.warm(&big).unwrap();
        assert!(pool.stats().resident_bytes > 0);
        let released = pool.release_all();
        assert!(released > 0);
        assert_eq!(pool.stats().resident_bytes, 0);
        assert_eq!(pool.stats().logical_bytes, 0);
        assert_eq!(pool.stats().arena_pages, 0);
    }
}
