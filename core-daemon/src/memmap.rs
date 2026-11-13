//! Page-aligned anonymous memory blocks that we fully own.
//!
//! Why not `Vec<u8>`? Because `madvise(MADV_DONTNEED)` is only safe on whole
//! pages that belong to a mapping we own. A `Vec` lives inside the malloc
//! arena: advising it away can corrupt allocator metadata. Every compressed
//! frame therefore lives in its own `mmap(MAP_ANON)` region, so evicting a
//! frame is a single `madvise` + `munmap` and the RSS really drops.

use crate::platform::{madvise_dont_need, madvise_will_need, page_size};
use std::io;
use std::ptr::NonNull;

fn round_up(n: usize, align: usize) -> usize {
    if align == 0 {
        return n;
    }
    (n + align - 1) / align * align
}

/// An owned, page-aligned, anonymous mapping.
pub struct PageBuf {
    ptr: NonNull<u8>,
    /// Usable length in bytes (page aligned).
    len: usize,
    /// Bytes actually written by the caller (logical content length).
    used: usize,
    /// Set after `dont_need()`: the pages are gone, content must not be read.
    dropped: bool,
}

// The mapping is exclusively owned by this struct (no aliasing), so moving
// between threads is safe.
unsafe impl Send for PageBuf {}
unsafe impl Sync for PageBuf {}

impl PageBuf {
    /// Allocate a zeroed block of at least `len` bytes.
    pub fn zeroed(len: usize) -> io::Result<Self> {
        let page = page_size();
        let mapped = round_up(len.max(1), page);
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                mapped,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            ptr: NonNull::new(ptr as *mut u8).expect("mmap returned null"),
            len: mapped,
            used: 0,
            dropped: false,
        })
    }

    /// Allocate and copy `data` into the block.
    pub fn from_slice(data: &[u8]) -> io::Result<Self> {
        let mut buf = Self::zeroed(data.len())?;
        buf.write(data);
        Ok(buf)
    }

    pub fn write(&mut self, data: &[u8]) {
        assert!(data.len() <= self.len, "PageBuf overflow");
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.as_ptr(), data.len());
        }
        self.used = data.len();
        self.dropped = false;
    }

    pub fn as_slice(&self) -> &[u8] {
        if self.dropped {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.used) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.used) }
    }

    /// Mapped (page-rounded) size — this is what counts against the RAM budget.
    pub fn mapped_len(&self) -> usize {
        self.len
    }

    pub fn used_len(&self) -> usize {
        self.used
    }

    pub fn is_dropped(&self) -> bool {
        self.dropped
    }

    /// Return the pages to XNU. RSS drops immediately; the mapping stays valid
    /// but reads zero-fill.
    pub fn dont_need(&mut self) -> io::Result<()> {
        if self.dropped {
            return Ok(());
        }
        unsafe { madvise_dont_need(self.ptr.as_ptr(), self.len)? };
        self.dropped = true;
        self.used = 0;
        Ok(())
    }

    /// Ask XNU to prefault the pages (used while warming from SSD).
    pub fn will_need(&self) -> io::Result<()> {
        if self.dropped {
            return Ok(());
        }
        unsafe { madvise_will_need(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for PageBuf {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr.as_ptr() as *mut libc::c_void, self.len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_read() {
        let mut b = PageBuf::zeroed(10).unwrap();
        b.write(b"hello");
        assert_eq!(b.as_slice(), b"hello");
        assert!(b.mapped_len() >= 10);
        assert_eq!(b.mapped_len() % page_size(), 0);
    }

    #[test]
    fn dont_need_drops_pages() {
        let mut b = PageBuf::from_slice(&vec![7u8; 100_000]).unwrap();
        let mapped = b.mapped_len();
        assert_eq!(b.as_slice().len(), 100_000);
        b.dont_need().unwrap();
        assert_eq!(b.as_slice().len(), 0);
        assert!(b.is_dropped());
        assert!(mapped > 0);
        // Second call is idempotent.
        b.dont_need().unwrap();
    }

    #[test]
    fn many_blocks() {
        let mut blocks: Vec<PageBuf> = (0..64)
            .map(|i| PageBuf::from_slice(&vec![i as u8; 4096]).unwrap())
            .collect();
        for (i, b) in blocks.iter().enumerate() {
            assert!(b.as_slice().iter().all(|v| *v == i as u8));
        }
        for b in blocks.iter_mut() {
            b.dont_need().unwrap();
        }
    }
}
