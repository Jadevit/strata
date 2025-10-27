use std::{path::PathBuf, process::Command};
use tauri::{AppHandle, Manager, path::BaseDirectory};

use super::discover::default_runtime_root;

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

/// Look up the sidecar installer (bundled or dev path).
pub(super) fn find_sidecar(app: &AppHandle) -> Option<PathBuf> {
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

/// Non-command helper used by the module-level tauri command.
pub(super) fn run_installer(
    prefer: Option<String>,
    manifest: Option<String>,
    app: &AppHandle,
) -> Result<(), String> {
    let exe = find_sidecar(app).ok_or_else(|| {
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
