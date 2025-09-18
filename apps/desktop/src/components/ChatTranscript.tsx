import React from "react";
import type { Message } from "../types";

// helpers (no replaceAll; use regex with /g)
const OPEN = "<think>";
const CLOSE = "</think>";
const stripThinkTags = (s: string) => s.replace(/<think>/g, "").replace(/<\/think>/g, "");

// Streaming-aware parser for <think> ... </think>
// - shows a Thoughts panel as soon as <think> appears (even if not closed yet)
// - hides raw tags from the visible assistant text while streaming
// - collapses duplicated nested "<think>" that may echo during generation
function extractThinkStreaming(s: string): {
  think?: string;
  visible: string;
  isOpen: boolean;
} {
  const openIdx = s.indexOf(OPEN);
  if (openIdx === -1) {
    // no think block; clean any stray tags defensively
    return { visible: stripThinkTags(s), isOpen: false };
  }

  const closeIdx = s.indexOf(CLOSE, openIdx + OPEN.length);

  if (closeIdx === -1) {
    // streaming: open without close
    let think = s.slice(openIdx + OPEN.length);
    // collapse nested open tags & remove any stray closing that sneaks in
    think = think.replace(/(?:<think>)+/g, "").replace(/<\/think>/g, "");
    const visiblePrefix = s.slice(0, openIdx);
    return {
      think: think.trim(),
      visible: stripThinkTags(visiblePrefix).trim(),
      isOpen: true,
    };
  }

  // complete block
  let think = s.slice(openIdx + OPEN.length, closeIdx);
  think = think.replace(/(?:<think>)+/g, "").replace(/<\/think>/g, "");

  const visibleRest = s.slice(closeIdx + CLOSE.length);
  const visible = stripThinkTags(s.slice(0, openIdx) + visibleRest);

  return {
    think: think.trim(),
    visible: visible.trim(),
    isOpen: false,
  };
}

function ChevronRight({ className = "h-3.5 w-3.5" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      <path d="M8 6l4 4-4 4" />
    </svg>
  );
}

export default function ChatTranscript({
  messages,
  endRef,
}: {
  messages: Message[];
  endRef: React.RefObject<HTMLDivElement>;
}) {
  const isEmpty = messages.length === 0;

  return (
    <main className="min-h-0 flex-1 overflow-y-auto px-4 py-4 pb-24 overscroll-contain">
      {isEmpty ? (
        <div className="grid min-h-[60vh] place-content-center text-center">
          <div className="text-[20px] font-semibold text-slate-300">Start a conversation.</div>
          <div className="mt-1 text-[16px] text-slate-400">
            Try: <span className="text-slate-300 italic">“Explain X like I’m five.”</span>
          </div>
        </div>
      ) : (
        <div className="mx-auto flex max-w-3xl flex-col gap-4">
          {messages.map((m, i) => {
            const aiRaw = m.ai ?? "";
            const { think, visible, isOpen } = extractThinkStreaming(aiRaw);

            return (
              <div key={i} className="flex flex-col gap-2">
                {/* user bubble */}
                {m.user && (
                  <div className="ml-auto max-w-[80%] rounded-2xl rounded-br-sm bg-gradient-to-b from-[#7A8CFF] to-[#5E73FF] px-3 py-2 text-sm text-white">
                    {m.user}
                  </div>
                )}

                {/* Thoughts (appears as soon as <think> streams in) */}
                {think && (
                  <details className="group max-w-[90%] text-xs text-slate-400" open={false}>
                    <summary className="flex cursor-pointer select-none items-center gap-1 hover:text-slate-300">
                      <ChevronRight className="h-3 w-3 transition-transform group-open:rotate-90" />
                      <span>Thoughts{isOpen ? " (streaming…)" : ""}</span>
                    </summary>
                    <pre className="mt-1 whitespace-pre-wrap break-words font-mono text-[11px] text-slate-400">
                      {think}
                    </pre>
                  </details>
                )}

                {/* assistant flat block (without raw <think> tags visible) */}
                {aiRaw && (
                  <div className="mr-auto max-w-[90%] text-[15px] leading-7 text-slate-100">
                    {visible}
                  </div>
                )}

                {i < messages.length - 1 && <div className="mt-1 h-px w-full bg-white/5" />}
              </div>
            );
          })}
          <div ref={endRef} />
        </div>
      )}
    </main>
  );
}