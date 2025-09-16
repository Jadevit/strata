// strata-core/src/debug.rs
#[cfg(feature = "utf8-trace")]
pub fn dump_str(label: &str, s: &str) {
    use std::fmt::Write;
    let mut hex = String::with_capacity(s.len() * 3);
    for b in s.as_bytes() {
        let _ = write!(&mut hex, "{:02X} ", b);
    }
    let mut cps = String::new();
    for ch in s.chars() {
        let _ = write!(&mut cps, "U+{:04X} ", ch as u32);
    }
    println!("🔎 [{label}] bytes: {hex}");
    println!("🔎 [{label}] cps  : {cps}");
}
#[cfg(not(feature = "utf8-trace"))]
pub fn dump_str(_label: &str, _s: &str) {}
