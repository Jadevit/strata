# Strata

[![CI](https://github.com/jadevit/strata/actions/workflows/ci.yml/badge.svg)](https://github.com/jadevit/strata/actions)
[![Release](https://img.shields.io/github/v/release/jadevit/strata)](https://github.com/jadevit/strata/releases)
[![License: AGPLv3](https://img.shields.io/badge/license-AGPLv3-blue.svg)](LICENSE)

**Strata** is a Rust-built, local-first AI platform focused on speed, modularity, and security.  
It provides a unified foundation for local inference, backend integration, and plugin-driven extensibility—all without cloud dependencies or Electron bloat.

## Features
- Native Rust backend with a custom FFI wrapper for `llama.cpp`
- Modular plugin system for extending Strata with new tools, models, or backends
- Cross-platform hardware profiler with smart runtime detection and caching
- Dynamic model registry that automatically parses and displays metadata
- Clean, responsive Tauri + React interface
- Designed for extensibility: image generation, TTS/STT, vision models, and more planned

## Installation
Download the latest release from the [Releases](https://github.com/jadevit/strata/releases) page.  
Strata will detect your system hardware and configure the correct runtime on first launch.

## Roadmap
- Plugin Store for community extensions  
- Additional backends (ONNX, MLC, Transformers)  
- Model benchmarking and performance analysis tools  
- Local fine-tuning pipeline (future milestone)  
- Sandbox support for secure remote runtimes  

## License
Strata is licensed under the **GNU Affero General Public License (AGPLv3)**.  
See the [LICENSE](LICENSE) and [NOTICE](NOTICE) files for details.

---

© 2025 Jaden Stanley.