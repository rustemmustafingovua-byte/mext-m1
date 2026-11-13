//! Packed arena for small compressed frames.
//!
//! A dedicated `mmap` per file is wasteful for tiny files: with 16 KiB pages a
//! 2 KiB LZ4 payload still costs 16 KiB of RSS. The arena packs many small
//! frames into shared pages inside one mapping, so sixteen 1 KiB frames need a
//! single page instead of sixteen.
//!
//! Reclaiming is still page-granular (that is what `madvise` offers), so pages
//! are reference counted: a frame eviction only calls `madvise(MADV_DONTNEED)`
//! when the *last* frame living on that page is gone. Every frame is immutable
//! after `alloc_copy`, so frames can share a page without aliasing hazards.

use crate::platform::{madvise_dont_need, page_size};
use std::io;
use std::ptr::NonNull;

/// Each chunk is a separate mapping; 4 MiB keeps the VMA count low while
/// avoiding a huge upfront reservation on an 8 GB machine.
pub const ARENA_CHUNK_BYTES: usize = 4 * 1024 * 1024;

/// Frames bigger than half a page get their own mapping (rounding them into the
/// arena would waste the same pages it is supposed to save).
const MAX_FRAME_FRACTION: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaHandle {
    chunk: u32,
    page: u32,
    offset: u32,
    len: u32,
}

struct Chunk {
    ptr: NonNull<u8>,
    pages: usize,
    /// Append pointer inside each page.
    used: Vec<u32>,
    /// Live frames per page; `0` means the page is reclaimable.
    refs: Vec<u16>,
}

pub struct SmallArena {
    page_size: usize,
    max_frame: usize,
    chunks: Vec<Chunk>,
    live_pages: u64,
    live_frames: u64,
    released: u64,
}

// The arena owns its mappings exclusively; frames only hold handles.
unsafe impl Send for SmallArena {}

impl SmallArena {
    pub fn new() -> Self {
        let page = page_size();
        Self {
            page_size: page,
            max_frame: page / MAX_FRAME_FRACTION,
            chunks: Vec::new(),
            live_pages: 0,
            live_frames: 0,
            released: 0,
        }
    }

    /// Largest payload the arena accepts; bigger frames need `PageBuf`.
    pub fn max_frame(&self) -> usize {
        self.max_frame
    }

    pub fn page_bytes(&self) -> u64 {
        self.page_size as u64
    }

    /// Copy `data` into the arena. Returns `Ok(None)` when the payload does not
    /// belong here (too large), so the caller can fall back to a dedicated map.
    pub fn alloc_copy(&mut self, data: &[u8]) -> io::Result<Option<ArenaHandle>> {
        if data.is_empty() || data.len() > self.max_frame {
            return Ok(None);
        }
        let len = data.len() as u32;

        // First fit over live pages.
        for (chunk_index, chunk) in self.chunks.iter_mut().enumerate() {
            for page in 0..chunk.pages {
                if chunk.used[page] as usize + data.len() <= self.page_size {
                    let offset = chunk.used[page];
                    chunk.used[page] += len;
                    if chunk.refs[page] == 0 {
                        self.live_pages += 1;
                    }
                    chunk.refs[page] = chunk.refs[page].saturating_add(1);
                    unsafe {
                        let dst = chunk
                            .ptr
                            .as_ptr()
                            .add(page * self.page_size + offset as usize);
                        std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
                    }
                    self.live_frames += 1;
                    return Ok(Some(ArenaHandle {
                        chunk: chunk_index as u32,
                        page: page as u32,
                        offset,
                        len,
                    }));
                }
            }
        }

        // No page has room: map a new chunk and use its first page.
        let index = self.add_chunk()?;
        let chunk = &mut self.chunks[index];
        chunk.used[0] = len;
        chunk.refs[0] = 1;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), chunk.ptr.as_ptr(), data.len());
        }
        self.live_pages += 1;
        self.live_frames += 1;
        Ok(Some(ArenaHandle {
            chunk: index as u32,
            page: 0,
            offset: 0,
            len,
        }))
    }

    fn add_chunk(&mut self) -> io::Result<usize> {
        let pages = ARENA_CHUNK_BYTES / self.page_size;
        let bytes = pages * self.page_size;
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        self.chunks.push(Chunk {
            ptr: NonNull::new(ptr as *mut u8).expect("mmap returned null"),
            pages,
            used: vec![0; pages],
            refs: vec![0; pages],
        });
        Ok(self.chunks.len() - 1)
    }

    /// Borrow the compressed payload of a live frame.
    pub fn view(&self, handle: &ArenaHandle) -> &[u8] {
        let chunk = &self.chunks[handle.chunk as usize];
        let base = handle.page as usize * self.page_size + handle.offset as usize;
        assert!(base + handle.len as usize <= chunk.pages * self.page_size);
        unsafe { std::slice::from_raw_parts(chunk.ptr.as_ptr().add(base), handle.len as usize) }
    }

    /// Drop a frame. Returns the number of bytes actually returned to the OS
    /// (non-zero only when the whole page became reclaimable).
    pub fn free(&mut self, handle: ArenaHandle) -> u64 {
        let page_size = self.page_size;
        let chunk = &mut self.chunks[handle.chunk as usize];
        let page = handle.page as usize;
        debug_assert!(chunk.refs[page] > 0);
        chunk.refs[page] = chunk.refs[page].saturating_sub(1);
        self.live_frames = self.live_frames.saturating_sub(1);
        if chunk.refs[page] > 0 {
            return 0; // neighbours are still alive: the page must stay
        }
        let address = unsafe { chunk.ptr.as_ptr().add(page * page_size) };
        let released = unsafe { madvise_dont_need(address, page_size) }.is_ok();
        chunk.used[page] = 0;
        self.live_pages = self.live_pages.saturating_sub(1);
        if released {
            self.released += page_size as u64;
            page_size as u64
        } else {
            0
        }
    }

    pub fn live_bytes(&self) -> u64 {
        self.live_pages * self.page_size as u64
    }

    pub fn live_pages(&self) -> u64 {
        self.live_pages
    }

    pub fn live_frames(&self) -> u64 {
        self.live_frames
    }

    pub fn released_bytes(&self) -> u64 {
        self.released
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }
}

impl Default for SmallArena {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for SmallArena {
    fn drop(&mut self) {
        for chunk in &self.chunks {
            unsafe {
                libc::munmap(
                    chunk.ptr.as_ptr() as *mut libc::c_void,
                    chunk.pages * self.page_size,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(len: usize, seed: u8) -> Vec<u8> {
        (0..len).map(|i| seed.wrapping_add(i as u8)).collect()
    }

    #[test]
    fn packs_small_frames_into_one_page() {
        let mut arena = SmallArena::new();
        let page = arena.page_bytes();
        let mut handles = Vec::new();
        // 8 × 1 KiB fits comfortably into one 16 KiB page.
        for i in 0..8 {
            let handle = arena.alloc_copy(&payload(1024, i as u8)).unwrap().unwrap();
            handles.push(handle);
        }
        assert!(
            arena.live_bytes() <= page,
            "live {} > page {}",
            arena.live_bytes(),
            page
        );
        assert_eq!(arena.live_pages(), 1);
        assert_eq!(arena.live_frames(), 8);
        for (i, handle) in handles.iter().enumerate() {
            assert_eq!(arena.view(handle), payload(1024, i as u8));
        }
    }

    #[test]
    fn reclaims_page_only_after_last_frame() {
        let mut arena = SmallArena::new();
        let page = arena.page_bytes();
        let a = arena.alloc_copy(&payload(2048, 1)).unwrap().unwrap();
        let b = arena.alloc_copy(&payload(2048, 2)).unwrap().unwrap();
        assert_eq!(arena.live_pages(), 1);
        assert_eq!(arena.free(a), 0, "page still holds frame b");
        assert_eq!(arena.live_pages(), 1);
        assert_eq!(arena.live_frames(), 1);
        assert_eq!(arena.free(b), page);
        assert_eq!(arena.live_pages(), 0);
        assert_eq!(arena.live_bytes(), 0);
        assert_eq!(arena.released_bytes(), page);
    }

    #[test]
    fn rejects_large_payloads() {
        let mut arena = SmallArena::new();
        let too_big = vec![0u8; arena.max_frame() + 1];
        assert!(arena.alloc_copy(&too_big).unwrap().is_none());
        assert!(arena.alloc_copy(&[]).unwrap().is_none());
        assert_eq!(arena.live_bytes(), 0);
    }

    #[test]
    fn grows_with_new_chunks() {
        let mut arena = SmallArena::new();
        let count = 4096; // 4096 × 2 KiB = 8 MiB -> at least two 4 MiB chunks
        let mut handles = Vec::new();
        for i in 0..count {
            handles.push(
                arena
                    .alloc_copy(&payload(2048, (i % 251) as u8))
                    .unwrap()
                    .unwrap(),
            );
        }
        assert!(arena.chunk_count() >= 2, "chunks {}", arena.chunk_count());
        assert_eq!(arena.live_frames(), count as u64);
        // Spot check integrity across chunk boundaries.
        for index in [0, 1024, 2048, 4095] {
            assert_eq!(
                arena.view(&handles[index]),
                payload(2048, (index % 251) as u8)
            );
        }
    }

    #[test]
    fn handles_reuse_after_free() {
        let mut arena = SmallArena::new();
        let first = arena.alloc_copy(&payload(1024, 7)).unwrap().unwrap();
        let page = arena.page_bytes();
        assert_eq!(arena.free(first), page);
        let second = arena.alloc_copy(&payload(1024, 9)).unwrap().unwrap();
        assert_eq!(arena.view(&second), payload(1024, 9));
        assert_eq!(arena.live_pages(), 1);
    }
}
