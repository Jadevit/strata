export interface Message {
  user: string;
  ai?: string;
}

export interface ModelEntry {
  id: string;
  name: string;
  path: string;
  backend_hint: string;
  file_type: string;
  family?: string;
}

export interface ModelMeta {
  name?: string;
  family?: string;
  backend: string;
  file_type: string;
  quantization?: string;
  context_length?: number;
  vocab_size?: number;
  eos_token_id?: number;
  bos_token_id?: number;
  prompt_flavor_hint?: "ChatMl" | "InstBlock" | "UserAssistant" | "Plain" | "Phi3";
  has_chat_template: boolean;
  raw?: Record<string, string>;
}

// src/types.ts
export type PreloadState = "idle" | "loading" | "ready" | "error";

export interface PreloadStatus {
  state: PreloadState;
  path?: string;
  error?: string | null;
}

export interface CpuInfo {
  brand: string;
  threads: number;
  avx2: boolean;
  avx512: boolean;
}

export interface GpuDriverInfo {
  cuda?: string | null;
  nvml?: string | null;
}

export interface GpuInfo {
  vendor_id: number;
  device_id: number;
  vendor: string;
  name: string;
  driver?: GpuDriverInfo | null;
}

export interface BackendSupport {
  cpu: boolean;
  cuda: boolean;
  rocm: boolean;
  vulkan: boolean;
  metal: boolean;
}

export interface HardwareProfile {
  schema: number;
  os: string;
  arch: string;
  cpu: CpuInfo;
  ram_gb: number;
  gpus: GpuInfo[];
  backends: BackendSupport;
  fingerprint: string;
  created_at: string;
  updated_at: string;
}