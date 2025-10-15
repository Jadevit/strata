use serde_json::Value as Json;
use std::{fs, path::PathBuf, process::Command};
use tauri::{AppHandle, Manager, path::BaseDirectory};

fn installer_exe_name() -> &'static str {
    #[cfg(windows)]
    {
        "runtime-installer.exe"
    }
    #[cfg(not(windows))]
    {
        "runtime-installer"
    }
}

pub fn default_runtime_root() -> Option<PathBuf> {
    // match the installer's per-user layout
    dirs::data_dir().map(|p| p.join("Strata").join("runtimes").join("llama"))
}

fn read_runtime_json(root: &std::path::Path) -> Option<Json> {
    let p = root.join("runtime.json");
    let bytes = std::fs::read(&p).ok()?;
    serde_json::from_slice::<Json>(&bytes).ok()
}

pub fn runtime_current_lib_dir(root: &std::path::Path) -> Option<std::path::PathBuf> {
    let j = read_runtime_json(root)?;
    j.get("llama")
        .and_then(|ll| ll.get("current_lib_dir"))
        .and_then(|v| v.as_str())
        .map(std::path::PathBuf::from)
}

pub fn runtime_is_monolith(root: &std::path::Path) -> bool {
    read_runtime_json(root)
        .and_then(|j| j.get("llama").cloned())
        .and_then(|ll| ll.get("monolith").cloned())
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

fn find_sidecar(app: &AppHandle) -> Option<PathBuf> {
    // 1) packaged app: bundled in Resources
    if let Some(p) = app
        .path()
        .resolve(installer_exe_name(), BaseDirectory::Resource)
        .ok()
    {
        if p.exists() {
            return Some(p);
        }
    }
    // 2) dev fallback
    let dev = PathBuf::from("apps/desktop/src-tauri/sidecar/runtime-installer/target/release")
        .join(installer_exe_name());
    if dev.exists() {
        return Some(dev);
    }
    None
}

#[tauri::command]
pub fn is_llama_runtime_installed() -> bool {
    if let Some(root) = default_runtime_root() {
        root.join("runtime.json").exists()
    } else {
        false
    }
}

#[tauri::command]
pub fn run_runtime_installer(
    prefer: Option<String>,
    manifest: Option<String>,
    app: AppHandle,
) -> Result<(), String> {
    let exe = find_sidecar(&app).ok_or_else(|| {
        "runtime-installer not found in Resources or local target; bundle it as a sidecar"
            .to_string()
    })?;

    let install_dir =
        default_runtime_root().ok_or_else(|| "could not resolve user data dir".to_string())?;

    let mut cmd = Command::new(&exe);
    cmd.arg("--install-dir").arg(install_dir);

    if let Some(p) = prefer {
        cmd.arg("--prefer").arg(p);
    }
    if let Some(m) = manifest {
        cmd.arg("--manifest").arg(m);
    }

    let status = cmd
        .status()
        .map_err(|e| format!("failed to spawn installer: {e}"))?;
    if !status.success() {
        return Err(format!(
            "installer exited with code {:?}",
            status.code().unwrap_or(-1)
        ));
    }
    Ok(())
}
