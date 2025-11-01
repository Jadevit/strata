#![allow(non_snake_case)]

use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use ash::vk;

pub fn env_timeout_ms() -> u64 {
    std::env::var("STRATA_HWPROF_TIMEOUT_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&v| v >= 200)
        .unwrap_or(2000)
}

pub fn hwprof_debug() -> bool {
    std::env::var("STRATA_HWPROF_DEBUG")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

pub fn disabled(name: &str) -> bool {
    let key = format!("STRATA_HWPROF_DISABLE_{}", name.to_ascii_uppercase());
    std::env::var(key)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Run a closure on a worker thread and join with timeout.
/// Returns (result, maybe_reason, elapsed_ms)
pub fn with_timeout<T: Send + 'static, F: FnOnce() -> T + Send + 'static>(
    label: &str,
    dur: Duration,
    f: F,
) -> (Option<T>, Option<String>, u64) {
    let (tx, rx) = mpsc::channel();
    let start = Instant::now();
    thread::spawn(move || {
        let out = f();
        let _ = tx.send(out);
    });
    let res = rx.recv_timeout(dur).ok();
    let ms = start.elapsed().as_millis() as u64;
    if res.is_none() {
        (None, Some(format!("{label}_timeout")), ms)
    } else {
        (res, None, ms)
    }
}

pub fn is_software_renderer(name: &str, dtype: vk::PhysicalDeviceType) -> Option<&'static str> {
    let lower = name.to_ascii_lowercase();
    if matches!(dtype, vk::PhysicalDeviceType::CPU) {
        return Some("cpu_adapter");
    }
    if lower.contains("llvmpipe") {
        return Some("llvmpipe");
    }
    if lower.contains("softpipe") {
        return Some("softpipe");
    }
    if lower.contains("swrast") {
        return Some("swrast");
    }
    if lower.contains("swiftshader") {
        return Some("swiftshader");
    }
    None
}

pub fn vram_from_heaps(props: &vk::PhysicalDeviceMemoryProperties) -> Option<u64> {
    let mut total: u128 = 0;
    for i in 0..props.memory_heap_count as usize {
        let heap = props.memory_heaps[i];
        if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
            total += heap.size as u128;
        }
    }
    if total > 0 {
        Some(total as u64)
    } else {
        None
    }
}

pub fn fmt_vk_driver(version: u32) -> String {
    // Not standardized across vendors; provide raw integer (hex) and parsed attempt.
    format!("{}", version)
}
