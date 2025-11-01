import { listen, UnlistenFn } from "@tauri-apps/api/event";
import type { HardwareProfile } from "../types";

export type StreamDeltaEvent = { delta: string };
export type StreamCompleteEvent = { text: string };

export function onLLMStream(handler: (delta: string) => void): Promise<UnlistenFn> {
  return listen<StreamDeltaEvent>("llm-stream", (e) => handler(e.payload?.delta ?? ""));
}

export function onLLMComplete(handler: (text: string) => void): Promise<UnlistenFn> {
  return listen<StreamCompleteEvent>("llm-complete", (e) => handler(e.payload?.text ?? ""));
}

// small helper to safely unlisten
export function safeUnlisten(un: UnlistenFn | null | undefined) {
  try {
    if (un) un();
  } catch {
    /* no-op */
  }
}

export const HWPROFILE_EVENT = "strata://hwprofile";

export function onHwProfile(cb: (p: HardwareProfile) => void) {
  // returns a Promise<UnlistenFn>
  return listen(HWPROFILE_EVENT, (e) => cb(e.payload as HardwareProfile));
}