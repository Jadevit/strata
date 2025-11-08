import { useEffect, useRef, useState } from "react";
import type { ModelEntry } from "../types";
import { Icons } from "../ui/icons";

function ChevronDown({ className = "h-3.5 w-3.5" }: { className?: string }) {
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
      <path d="M6 8l4 4 4-4" />
    </svg>
  );
}

function BackendPill({ backend }: { backend: string }) {
  const label = backend === "llama" ? "Llama" : backend.charAt(0).toUpperCase() + backend.slice(1);
  return (
    <span className="ml-2 shrink-0 rounded-full border border-white/15 px-2 py-[2px] text-[10px] text-slate-300">
      {label}
    </span>
  );
}

export default function Header({
  selectedModel,
  recent,
  onSelectModel,
  onOpenInfo,
  onImportModel,
}: {
  selectedModel: ModelEntry | null;
  recent: ModelEntry[];
  onSelectModel: (m: ModelEntry) => void;
  onOpenInfo: () => void;
  onImportModel: () => void | Promise<void>;
}) {
  const [modelOpen, setModelOpen] = useState(false);
  const [kebabOpen, setKebabOpen] = useState(false);

  const modelBtnRef = useRef<HTMLButtonElement | null>(null);
  const modelMenuRef = useRef<HTMLDivElement | null>(null);
  const kebabBtnRef = useRef<HTMLButtonElement | null>(null);
  const kebabMenuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    function onDocMouseDown(e: MouseEvent) {
      const t = e.target as Node;
      if (modelOpen) {
        const insideModel = modelMenuRef.current?.contains(t) || modelBtnRef.current?.contains(t);
        if (!insideModel) setModelOpen(false);
      }
      if (kebabOpen) {
        const insideKebab = kebabMenuRef.current?.contains(t) || kebabBtnRef.current?.contains(t);
        if (!insideKebab) setKebabOpen(false);
      }
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") {
        setModelOpen(false);
        setKebabOpen(false);
      }
    }
    document.addEventListener("mousedown", onDocMouseDown);
    window.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDocMouseDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [modelOpen, kebabOpen]);

  return (
    <header className="sticky top-0 z-20 grid grid-cols-[1fr_auto_1fr] items-center border-b border-white/10 px-4 py-3">
      {/* LEFT: logo only (nav is always-compact now) */}
      <div className="flex items-center gap-3">
        <div
          className="h-7 w-7 select-none rounded-md bg-white/5 text-center text-[12px] leading-[28px] text-slate-300"
          title="Logo"
          aria-label="Logo"
        >
          S
        </div>
      </div>

      {/* CENTER: MRU dropdown (unchanged) */}
      <div className="justify-self-center">
        <div className="relative">
          <button
            ref={modelBtnRef}
            className="group inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-sm text-slate-200 hover:bg-white/10 focus:outline-none focus:ring-4 focus:ring-white/10"
            onClick={() => {
              setModelOpen((v) => !v);
              setKebabOpen(false);
            }}
            aria-haspopup="menu"
            aria-expanded={modelOpen}
            title="Select model"
          >
            <span className="text-slate-400">Model:</span>
            <span className="truncate text-slate-100 max-w-[24ch]">
              {selectedModel ? selectedModel.name : "Select…"}
            </span>
            <ChevronDown className={`h-3.5 w-3.5 transition-transform ${modelOpen ? "rotate-180" : ""} text-slate-300`} />
          </button>

          {modelOpen && (
            <div
              ref={modelMenuRef}
              className="absolute left-1/2 top-[110%] z-50 w-[20rem] -translate-x-1/2 overflow-hidden rounded-lg border border-white/10 bg-[#0B0F1A] shadow"
              role="menu"
            >
              {recent.length > 0 ? (
                recent.map((m) => (
                  <button
                    key={m.id}
                    className="flex w-full items-center px-3 py-2 text-left hover:bg-white/5"
                    onClick={() => {
                      onSelectModel(m);
                      setModelOpen(false);
                    }}
                    role="menuitem"
                    title={m.path}
                  >
                    <span className="min-w-0 truncate text-sm text-slate-100">{m.name}</span>
                    <BackendPill backend={m.backend_hint} />
                  </button>
                ))
              ) : (
                <div className="px-3 py-2 text-sm text-slate-400">No recent models</div>
              )}

              <div className="my-1 h-px bg-white/10" />

              <button
                className="grid w-full place-items-center px-3 py-3 text-[12px] text-slate-200 hover:bg-white/5"
                onClick={() => {
                  void onImportModel();
                  setModelOpen(false);
                }}
                role="menuitem"
                title="Import Model"
              >
                <span className="opacity-90">Import Model</span>
              </button>
            </div>
          )}
        </div>
      </div>

      {/* RIGHT: info + kebab */}
      <div className="relative flex items-center justify-self-end gap-2">
        <button
          className="rounded-lg px-2 py-1.5 text-sm text-slate-200 hover:bg-white/10 focus:outline-none focus:ring-4 focus:ring-white/10"
          onClick={() => {
            setKebabOpen(false);
            onOpenInfo();
          }}
          aria-label="Open model info"
          title="Model info"
        >
          <Icons.Info />
        </button>

        <div className="relative">
          <button
            ref={kebabBtnRef}
            className="rounded-lg px-2 py-1.5 text-sm text-slate-200 hover:bg-white/10 focus:outline-none focus:ring-4 focus:ring-white/10"
            onClick={() => {
              setKebabOpen((v) => !v);
              setModelOpen(false);
            }}
            aria-haspopup="menu"
            aria-expanded={kebabOpen}
            title="More"
          >
            <Icons.More />
          </button>

          {kebabOpen && (
            <div
              ref={kebabMenuRef}
              className="absolute right-0 top-[110%] z-50 w-56 overflow-hidden rounded-lg border border-white/10 bg-[#0B0F1A] shadow"
              role="menu"
            >
              <button
                className="flex w-full items-center justify-between px-3 py-2 text-left text-sm opacity-60 cursor-not-allowed"
                aria-disabled="true"
                title="Coming soon"
              >
                Theme <span className="text-xs text-slate-400">Dark</span>
              </button>
              <button
                className="flex w-full items-center justify-between px-3 py-2 text-left text-sm opacity-60 cursor-not-allowed"
                aria-disabled="true"
                title="Coming soon"
              >
                Export chat <span className="text-xs text-slate-400">MD / JSON</span>
              </button>
              <button
                className="flex w-full items-center justify-between px-3 py-2 text-left text-sm opacity-60 cursor-not-allowed"
                aria-disabled="true"
                title="Coming soon"
              >
                Settings
              </button>
              <div className="my-1 h-px bg-white/10" />
              <button
                className="flex w-full items-center justify-between px-3 py-2 text-left text-sm opacity-60 cursor-not-allowed"
                aria-disabled="true"
                title="Coming soon"
              >
                Clear chat
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}