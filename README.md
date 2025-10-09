# Strata

[![CI](https://github.com/jadevit/strata/actions/workflows/ci.yml/badge.svg)](https://github.com/jadevit/strata/actions)
[![Release](https://img.shields.io/github/v/release/jadevit/strata)](https://github.com/jadevit/strata/releases)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![License: Commercial](https://img.shields.io/badge/license-Commercial-orange.svg)](LICENSE-COMMERCIAL)

Strata is a local-first AI platform built with Rust, Tauri, and React.  
It delivers fast inference, model management, and hardware-aware runtimes — without Electron bloat or external dependencies.

## Features
- Custom Rust inference loop with a handwritten FFI wrapper for `llama.cpp`
- GPU runtimes (CUDA, Vulkan, Metal) with automatic installer detection
- Multiple backend architecture (currently supports `llama.cpp`, designed for ONNX, Transformers, and more)
- Model manager with metadata scraping and dynamic prompt formatting
- Built-in benchmarking system with reproducible `.sbx` files
- Logging system for conversations and model runs
- CI/CD pipeline for cross-platform runtime publishing

## Installation
Download the latest release from [Releases](https://github.com/jadevit/strata/releases).  
The installer detects your hardware and installs the correct runtime automatically.

## Roadmap
- Fine-tuning pipeline
- Additional backends (ONNX, Transformers, etc.)
- Remote inference with sandboxing and auditing
- Offline mode with strict security guardrails

## License
Strata is dual-licensed:

- **Apache 2.0** for personal, research, and open-source use  
- **Commercial license** for companies and organizations using Strata in internal systems, products, or services  

For commercial inquiries, contact: [your email goes here].