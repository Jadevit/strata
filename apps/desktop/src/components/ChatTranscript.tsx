import React from "react";
import type { Message } from "../types";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";

// ----- <think> helpers -----
const OPEN = "<think>";
const CLOSE = "</think>";
const stripThinkTags = (s: string) => s.replace(/<think>/g, "").replace(/<\/think>/g, "");

function extractThinkStreaming(s: string): {
  think?: string;
  visible: string;
  isOpen: boolean;
} {
  const openIdx = s.indexOf(OPEN);
  if (openIdx === -1) return { visible: stripThinkTags(s), isOpen: false };

  const closeIdx = s.indexOf(CLOSE, openIdx + OPEN.length);
  if (closeIdx === -1) {
    let think = s.slice(openIdx + OPEN.length);
    think = think.replace(/(?:<think>)+/g, "").replace(/<\/think>/g, "");
    const visiblePrefix = s.slice(0, openIdx);
    return {
      think: think.trim(),
      visible: stripThinkTags(visiblePrefix).trim(),
      isOpen: true,
    };
  }

  let think = s.slice(openIdx + OPEN.length, closeIdx);
  think = think.replace(/(?:<think>)+/g, "").replace(/<\/think>/g, "");
  const visibleRest = s.slice(closeIdx + CLOSE.length);
  const visible = stripThinkTags(s.slice(0, openIdx) + visibleRest);
  return { think: think.trim(), visible: visible.trim(), isOpen: false };
}

// ----- UI bits -----
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

// ----- Code block with copy button -----
type CodeBlockProps = {
  inline?: boolean;
  className?: string;
  children?: React.ReactNode;
};

const CodeBlock: React.FC<CodeBlockProps> = ({ inline, className, children }) => {
  const match = /language-(\w+)/.exec(className || "");
  const code = String(children ?? "").replace(/\n$/, "");

  if (inline) {
    return (
      <code className="rounded bg-white/10 px-1 py-0.5 font-mono text-[0.85em]">
        {children}
      </code>
    );
  }

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
    } catch {
      /* ignore */
    }
  };

  return (
    <div className="group relative my-3 overflow-hidden rounded-lg border border-white/10">
      <button
        onClick={handleCopy}
        className="absolute right-2 top-2 hidden rounded-md bg-white/10 px-2 py-1 text-xs text-slate-200 backdrop-blur-sm transition group-hover:block"
        title="Copy"
      >
        Copy
      </button>
      <SyntaxHighlighter
        language={match?.[1]}
        style={vscDarkPlus}
        PreTag="div"
        showLineNumbers
        customStyle={{
          margin: 0,
          background: "transparent",
          fontSize: "0.95em",
        }}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
};

// Map Markdown → React
const mdComponents = {
  code: (props: any) => {
    const { inline, className, children } = props as CodeBlockProps;
    return (
      <CodeBlock inline={inline} className={className}>
        {children}
      </CodeBlock>
    );
  },
  a: (props: any) => (
    <a {...props} className="underline decoration-slate-500 hover:decoration-slate-300" />
  ),
  table: (props: any) => (
    <div className="not-prose overflow-x-auto">
      <table {...props} />
    </div>
  ),
};

// ----- Main transcript -----
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

                {/* Thoughts drawer */}
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

                {/* assistant message w/ Markdown */}
                {aiRaw && (
                  <article className="mr-auto max-w-[90%] prose prose-invert prose-pre:my-0 prose-pre:bg-transparent prose-code:before:hidden prose-code:after:hidden">
                    <ReactMarkdown remarkPlugins={[remarkGfm]} components={mdComponents}>
                      {visible}
                    </ReactMarkdown>
                  </article>
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