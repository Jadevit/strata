import { useCallback, useRef, useState } from "react";
import type { Message } from "../types";
import { runLLM, runLLMStream, cancelGeneration } from "../lib/api";
import { onLLMStream, onLLMComplete, safeUnlisten } from "../lib/events";
import type { UnlistenFn } from "@tauri-apps/api/event";

export function useLLM(selectedModelId?: string | null) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [streamingEnabled, setStreamingEnabled] = useState(true);

  // keep current unlisten fns so we can cancel safely on error
  const unlistenStreamRef = useRef<UnlistenFn | null>(null);
  const unlistenDoneRef = useRef<UnlistenFn | null>(null);

  // Trim overlap between previous text and the new delta.
  // Fixes doubled tokens like: "<think><think>Okay Okay so so ..."
  function trimOverlap(prev: string, delta: string, maxWindow = 200): string {
    if (!prev || !delta) return delta;
    const window = prev.slice(-maxWindow);
    const maxLen = Math.min(window.length, delta.length);
    for (let len = maxLen; len > 0; len--) {
      if (window.slice(-len) === delta.slice(0, len)) {
        return delta.slice(len);
      }
    }
    return delta;
  }

  const appendDeltaToLast = useCallback((incoming: string) => {
    if (!incoming) return;
    setMessages((prev) => {
      if (prev.length === 0) return prev;
      const out = [...prev];
      const last = out[out.length - 1];
      if (!last) return out;

      const current = last.ai || "";

      // Overlap guard
      let delta = trimOverlap(current, incoming);

      // (defensive) collapse immediate duplicated think tags in the delta chunk itself
      // without touching the final text semantics; UI parser also handles this.
      delta = delta.replace(/(?:<think>){2,}/g, "<think>").replace(/(?:<\/think>){2,}/g, "</think>");

      if (!delta) return out;
      out[out.length - 1] = { ...last, ai: current + delta };
      return out;
    });
  }, []);

  const sendMessage = useCallback(async () => {
    const prompt = input.trim();
    if (!prompt || isGenerating) return;

    // seed transcript
    setMessages((prev) => [...prev, { user: prompt, ai: streamingEnabled ? "" : "Typing..." }]);
    setInput("");
    setIsGenerating(true);

    if (!streamingEnabled) {
      try {
        const response = await runLLM(prompt, selectedModelId ?? null);
        setMessages((prev) => {
          const out = [...prev];
          const last = out[out.length - 1];
          if (last) out[out.length - 1] = { ...last, ai: response };
          return out;
        });
      } catch (err) {
        setMessages((prev) => {
          if (prev.length === 0) return prev;
          const out = [...prev];
          const last = out[out.length - 1];
          if (last) out[out.length - 1] = { ...last, ai: `[ERROR] ${String(err)}` };
          return out;
        });
      } finally {
        setIsGenerating(false);
      }
      return;
    }

    // streaming path
    try {
      // subscribe BEFORE invoking to avoid missing early tokens
      unlistenStreamRef.current = await onLLMStream((delta) => appendDeltaToLast(delta));
      unlistenDoneRef.current = await onLLMComplete((finalText) => {
        if (finalText) {
          setMessages((prev) => {
            if (prev.length === 0) return prev;
            const out = [...prev];
            const last = out[out.length - 1];
            if (!last) return out;
            out[out.length - 1] = { ...last, ai: finalText };
            return out;
          });
        }
        setIsGenerating(false);
        safeUnlisten(unlistenStreamRef.current);
        safeUnlisten(unlistenDoneRef.current);
        unlistenStreamRef.current = null;
        unlistenDoneRef.current = null;
      });

      await runLLMStream(prompt, selectedModelId ?? null);
    } catch (err) {
      setMessages((prev) => {
        if (prev.length === 0) return prev;
        const out = [...prev];
        const last = out[out.length - 1];
        if (!last) return out;
        out[out.length - 1] = { ...last, ai: `[ERROR] ${String(err)}` };
        return out;
      });
      setIsGenerating(false);
      safeUnlisten(unlistenStreamRef.current);
      safeUnlisten(unlistenDoneRef.current);
      unlistenStreamRef.current = null;
      unlistenDoneRef.current = null;
    }
  }, [appendDeltaToLast, input, isGenerating, streamingEnabled, selectedModelId]);

  const stop = useCallback(async () => {
    try {
      await cancelGeneration();
      // completion handler will fire with whatever text we have
    } catch (err) {
      console.error("cancel_generation failed:", err);
    }
  }, []);

  const newChat = useCallback(() => {
    setMessages([]);
    setInput("");
  }, []);

  return {
    messages,
    setMessages,
    input,
    setInput,
    isGenerating,
    setIsGenerating,
    streamingEnabled,
    setStreamingEnabled,
    sendMessage,
    stop,
    newChat,
  };
}