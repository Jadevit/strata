import { invoke } from "@tauri-apps/api/core";
import type { ModelEntry, ModelMeta } from "../types";

export type MetaIndexState = "idle" | "loading" | "ready" | "error";
export interface MetaIndexStatus {
  state: MetaIndexState;
  total: number;
  done: number;
  error: string | null;
}

// ---------- System prompt ----------
export async function loadSystemPrompt(): Promise<string> {
  return invoke<string>("load_system_prompt");
}

// ---------- Models ----------
export async function getModelList(): Promise<ModelEntry[]> {
  return invoke<ModelEntry[]>("get_model_list");
}

export async function getActiveModel(): Promise<string | null> {
  return invoke<string | null>("get_active_model");
}

export async function setActiveModelCmd(name: string): Promise<void> {
  return invoke("set_active_model_cmd", { name });
}

export async function getModelsRoot(): Promise<string> {
  return invoke<string>("get_models_root");
}

// Import a model file into the user library (copies the file).
export async function importModel(srcPath: string, family?: string): Promise<ModelEntry> {
  return invoke<ModelEntry>("import_model", { srcPath, family: family ?? null });
}

// ---------- Metadata (single file) ----------
export async function getModelMetadata(): Promise<ModelMeta> {
  return invoke<ModelMeta>("get_model_metadata");
}

// ---------- Metadata indexer (background cache) ----------
export async function metaStartIndex(force = false): Promise<void> {
  return invoke("meta_start_index", { force });
}

export async function metaStatus(): Promise<MetaIndexStatus> {
  return invoke<MetaIndexStatus>("meta_status");
}

export async function metaGetCached(id: string): Promise<ModelMeta | null> {
  return invoke<ModelMeta | null>("meta_get_cached", { id });
}

export async function metaClear(): Promise<void> {
  return invoke("meta_clear");
}

// ---------- LLM ----------
export async function runLLM(prompt: string, modelId?: string | null): Promise<string> {
  return invoke<string>("run_llm", {
    prompt,
    tts: false,
    model_id: modelId ?? null,
  });
}

export async function runLLMStream(prompt: string, modelId?: string | null): Promise<void> {
  return invoke("run_llm_stream", {
    prompt,
    tts: false,
    model_id: modelId ?? null,
  });
}

export async function cancelGeneration(): Promise<void> {
  return invoke("cancel_generation");
}