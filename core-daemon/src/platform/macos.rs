//! macOS-specific primitives: `sysctl`, `vm_stat` parsing, memory-pressure
//! level, swap usage and `madvise(2)`.

use anyhow::{Context, Result};
use serde::Serialize;
use std::ffi::CString;
use std::process::Command;

// ---------------------------------------------------------------------------
// sysctl
// ---------------------------------------------------------------------------

/// `sysctlbyname` for 64-bit integer values.
pub fn sysctl_u64(name: &str) -> Result<u64> {
    let cname = CString::new(name)?;
    let mut value: u64 = 0;
    let mut size = std::mem::size_of::<u64>();
    let rc = unsafe {
        libc::sysctlbyname(
            cname.as_ptr(),
            &mut value as *mut u64 as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return Err(std::io::Error::last_os_error()).with_context(|| format!("sysctl {name}"));
    }
    Ok(value)
}

/// `sysctlbyname` for 32-bit integer values.
pub fn sysctl_u32(name: &str) -> Result<u32> {
    let cname = CString::new(name)?;
    let mut value: u32 = 0;
    let mut size = std::mem::size_of::<u32>();
    let rc = unsafe {
        libc::sysctlbyname(
            cname.as_ptr(),
            &mut value as *mut u32 as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return Err(std::io::Error::last_os_error()).with_context(|| format!("sysctl {name}"));
    }
    Ok(value)
}

/// `sysctlbyname` for strings (`sysctl -n ...` equivalent).
pub fn sysctl_string(name: &str) -> Result<String> {
    let cname = CString::new(name)?;
    let mut size: usize = 0;
    let rc = unsafe {
        libc::sysctlbyname(
            cname.as_ptr(),
            std::ptr::null_mut(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 || size == 0 {
        return Err(std::io::Error::last_os_error()).with_context(|| format!("sysctl {name}"));
    }
    let mut buf = vec![0u8; size];
    let rc = unsafe {
        libc::sysctlbyname(
            cname.as_ptr(),
            buf.as_mut_ptr() as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return Err(std::io::Error::last_os_error()).with_context(|| format!("sysctl {name}"));
    }
    while buf.last() == Some(&0) {
        buf.pop();
    }
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

// ---------------------------------------------------------------------------
// VM statistics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct MemoryInfo {
    /// Physical installed RAM.
    pub total: u64,
    /// Page size used by XNU (16 KiB on Apple Silicon).
    pub page_size: u64,
    /// Completely free pages.
    pub free: u64,
    /// Pages holding recently used file-backed data (reclaimable).
    pub cached: u64,
    /// Anonymous application memory (approximates the macOS "App Memory" bar).
    pub app: u64,
    /// Wired kernel memory, cannot be paged out.
    pub wired: u64,
    /// Physical bytes currently occupied by the compressor.
    pub compressor_occupied: u64,
    /// Logical size of data stored compressed.
    pub compressor_data: u64,
    pub purgeable: u64,
    pub speculative: u64,
    pub anonymous: u64,
}

impl MemoryInfo {
    /// Bytes the user/kernel can hand out without pressure.
    pub fn available(&self) -> u64 {
        self.free.saturating_add(self.cached)
    }

    /// 0.0 (idle) .. 1.0 (no headroom).
    pub fn used_ratio(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        (self.total.saturating_sub(self.available())) as f64 / self.total as f64
    }
}

/// Parse `vm_stat` output. We shell out because the `host_statistics64`
/// struct layout is version-dependent and `vm_stat` is stable across macOS
/// releases; the call costs ~4 ms and is cached by the collector.
pub fn memory_info() -> Result<MemoryInfo> {
    let page_size = sysctl_u64("hw.pagesize").unwrap_or(16384);
    let total = sysctl_u64("hw.memsize").unwrap_or(0);
    let out = Command::new("vm_stat")
        .output()
        .context("running vm_stat")?;
    let text = String::from_utf8_lossy(&out.stdout);

    let get = |label: &str| -> u64 {
        text.lines()
            .find(|l| l.trim_start().starts_with(label))
            .and_then(|l| l.split(':').nth(1))
            .map(|v| {
                v.trim()
                    .trim_end_matches('.')
                    .chars()
                    .filter(|c| c.is_ascii_digit())
                    .collect::<String>()
            })
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0)
    };

    let free_pages = get("Pages free");
    let active = get("Pages active");
    let inactive = get("Pages inactive");
    let speculative = get("Pages speculative");
    let purgeable = get("Pages purgeable");
    let wired = get("Pages wired down");
    let anonymous = get("Anonymous pages");
    let filebacked = get("File-backed pages");
    let comp_pages = get("Pages stored in compressor");
    let comp_occupied = get("Pages occupied by compressor");

    let p = page_size as u128;
    let bytes = |pages: u64| -> u64 { (pages as u128 * p) as u64 };

    let app = bytes(anonymous)
        .saturating_sub(bytes(purgeable))
        .saturating_add(bytes(active).saturating_sub(bytes(filebacked)) / 4);

    Ok(MemoryInfo {
        total,
        page_size,
        free: bytes(free_pages),
        cached: bytes(inactive).saturating_add(bytes(speculative)),
        app,
        wired: bytes(wired),
        compressor_occupied: bytes(comp_occupied),
        compressor_data: bytes(comp_pages),
        purgeable: bytes(purgeable),
        speculative: bytes(speculative),
        anonymous: bytes(anonymous),
    })
}

// ---------------------------------------------------------------------------
// Memory pressure + swap
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Pressure {
    Green,
    Yellow,
    Red,
    Unknown,
}

impl Pressure {
    pub fn as_str(self) -> &'static str {
        match self {
            Pressure::Green => "green",
            Pressure::Yellow => "yellow",
            Pressure::Red => "red",
            Pressure::Unknown => "unknown",
        }
    }

    /// How aggressively the cache should be shrunk (1.0 = no shrink).
    pub fn shrink_factor(self) -> f64 {
        match self {
            Pressure::Green => 1.0,
            Pressure::Yellow => 0.7,
            Pressure::Red => 0.4,
            Pressure::Unknown => 0.9,
        }
    }
}

/// `kern.memorystatus_vm_pressure_level`: 1 = normal, 2 = warning, 4 = critical.
/// Falls back to a heuristic on `vm_stat` if the sysctl is unavailable.
pub fn memory_pressure(info: Option<&MemoryInfo>) -> Pressure {
    if let Ok(level) = sysctl_u32("kern.memorystatus_vm_pressure_level") {
        match level {
            0 | 1 => return Pressure::Green,
            2 => return Pressure::Yellow,
            4 => return Pressure::Red,
            _ => {}
        }
    }
    match info {
        Some(i) if i.total > 0 => {
            let used = i.used_ratio();
            if used > 0.94 {
                Pressure::Red
            } else if used > 0.85 {
                Pressure::Yellow
            } else {
                Pressure::Green
            }
        }
        _ => Pressure::Unknown,
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct SwapInfo {
    pub total: u64,
    pub used: u64,
}

/// Parse `sysctl vm.swapusage`:
/// `total = 1024.00M  used = 124.00M  free = 900.00M  (encrypted)`
pub fn swap_usage() -> SwapInfo {
    let mut info = match sysctl_string("vm.swapusage") {
        Ok(text) => parse_swap_text(&text),
        Err(_) => SwapInfo::default(),
    };
    // `vm.swapusage` is not reachable through `sysctlbyname` on every macOS
    // release; `sysctl -n` (the classic name→MIB path) always works.
    if info.total == 0 {
        if let Ok(out) = Command::new("sysctl").args(["-n", "vm.swapusage"]).output() {
            if out.status.success() {
                let fallback = parse_swap_text(&String::from_utf8_lossy(&out.stdout));
                if fallback.total > 0 {
                    info = fallback;
                }
            }
        }
    }
    info
}

/// `total = 2048.00M  used = 807.12M  free = 1240.88M  (encrypted)`
pub fn parse_swap_text(text: &str) -> SwapInfo {
    let normalized = text
        .replace(" = ", "=")
        .replace("= ", "=")
        .replace(" =", "=");
    let mut info = SwapInfo::default();
    for part in normalized.split_whitespace() {
        let Some((key, val)) = part.split_once('=') else {
            continue;
        };
        let target = match key.trim() {
            "total" => &mut info.total,
            "used" => &mut info.used,
            _ => continue,
        };
        *target = parse_scaled(val).unwrap_or(0);
    }
    info
}

fn parse_scaled(raw: &str) -> Option<u64> {
    let raw = raw.trim();
    let (num, unit) = raw.split_at(
        raw.char_indices()
            .find(|(_, c)| !c.is_ascii_digit() && *c != '.')
            .map(|(i, _)| i)
            .unwrap_or(raw.len()),
    );
    let value: f64 = num.parse().ok()?;
    let mult = match unit.trim().to_ascii_uppercase().as_str() {
        "" | "B" => 1.0,
        "K" => 1024.0,
        "M" => 1024.0 * 1024.0,
        "G" => 1024.0 * 1024.0 * 1024.0,
        _ => return None,
    };
    Some((value * mult) as u64)
}

// ---------------------------------------------------------------------------
// madvise(2) — the magic that actually hands pages back to XNU
// ---------------------------------------------------------------------------

/// Page size of this machine.
pub fn page_size() -> usize {
    sysctl_u64("hw.pagesize").unwrap_or(4096).max(4096) as usize
}

/// Drop the given range: XNU discards the pages immediately and the RSS of the
/// process decreases right away (unlike `free()`, which usually keeps the pages
/// in the malloc arena).
///
/// # Safety
/// `ptr` must point to a mapping owned by the caller, `len` may be any length
/// inside that mapping.
pub unsafe fn madvise_dont_need(ptr: *mut u8, len: usize) -> std::io::Result<()> {
    if ptr.is_null() || len == 0 {
        return Ok(());
    }
    let rc = libc::madvise(ptr as *mut libc::c_void, len, libc::MADV_DONTNEED);
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// Hint the kernel to fault a range in ahead of the first access.
///
/// # Safety
/// `ptr`/`len` must describe a valid mapping.
pub unsafe fn madvise_will_need(ptr: *mut u8, len: usize) -> std::io::Result<()> {
    if ptr.is_null() || len == 0 {
        return Ok(());
    }
    let rc = libc::madvise(ptr as *mut libc::c_void, len, libc::MADV_WILLNEED);
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_swap_scaled() {
        assert_eq!(parse_scaled("1024.00M"), Some(1024 * 1024 * 1024));
        assert_eq!(parse_scaled("512K"), Some(512 * 1024));
        assert_eq!(parse_scaled("2.00G"), Some(2 * 1024 * 1024 * 1024));
        assert_eq!(parse_scaled("0B"), Some(0));
    }

    #[test]
    fn parse_swap_line() {
        let info =
            parse_swap_text("total = 2048.00M  used = 807.12M  free = 1240.88M  (encrypted)");
        assert_eq!(info.total, 2048 * 1024 * 1024);
        assert_eq!(info.used, (807.12 * 1024.0 * 1024.0) as u64);
        let zero = parse_swap_text("total = 0B  used = 0B  free = 0B");
        assert_eq!(zero.total, 0);
        assert_eq!(zero.used, 0);
    }

    #[test]
    fn pressure_has_factor() {
        assert_eq!(Pressure::Green.shrink_factor(), 1.0);
        assert!(Pressure::Red.shrink_factor() < Pressure::Yellow.shrink_factor());
    }

    #[test]
    fn reads_machine_facts() {
        let total = sysctl_u64("hw.memsize").expect("hw.memsize");
        assert!(total > 1024 * 1024 * 1024, "total RAM = {total}");
        assert!(page_size() >= 4096);
    }

    #[test]
    fn reads_vm_stat() {
        let info = memory_info().expect("vm_stat");
        assert!(info.total > 0);
        assert!(info.free > 0);
        assert!(info.available() <= info.total + info.cached);
    }
}
