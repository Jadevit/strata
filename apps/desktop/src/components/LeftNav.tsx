import React from "react";
import { Icons } from "../ui/icons";

type Item = {
  key: string;
  label: string;
  icon: React.ReactNode;
  disabled?: boolean;
  onClick?: () => void;
};

export default function LeftNav({
  open,
  active = "chat",
  onNewChat,
  onOpenModels,
  onRequestOpen,
}: {
  open: boolean;
  active?: string;
  onNewChat: () => void;
  onOpenModels: () => void;
  onRequestOpen: () => void;
}) {
  const click = (fn?: () => void) => () => {
    if (!open) {
      onRequestOpen();
      requestAnimationFrame(() => fn && fn());
    } else {
      fn && fn();
    }
  };

  const ITEMS: Item[] = [
    { key: "new", label: "New Chat", icon: <Icons.New className="text-emerald-300 drop-shadow-[0_0_6px_rgba(16,185,129,0.35)]" />, onClick: onNewChat },
    { key: "chat", label: "Chat", icon: <Icons.Chat className="text-sky-300 drop-shadow-[0_0_6px_rgba(125,211,252,0.35)]" /> },
    { key: "models", label: "Models", icon: <Icons.Models className="text-violet-300 drop-shadow-[0_0_6px_rgba(167,139,250,0.35)]" />, onClick: onOpenModels },
    { key: "train", label: "Train", icon: <Icons.Train className="text-amber-300 drop-shadow-[0_0_6px_rgba(251,191,36,0.35)]" />, disabled: true },
    { key: "library", label: "Library", icon: <Icons.Library className="text-indigo-300 drop-shadow-[0_0_6px_rgba(99,102,241,0.35)]" />, disabled: true },
    { key: "results", label: "Results", icon: <Icons.Results className="text-green-300 drop-shadow-[0_0_6px_rgba(134,239,172,0.35)]" />, disabled: true },
    { key: "logs", label: "Logs", icon: <Icons.Logs className="text-rose-300 drop-shadow-[0_0_6px_rgba(244,63,94,0.35)]" />, disabled: true },
  ];

  return (
    <aside
      className={[
        "shrink-0 border-r border-white/10 bg-[#0B0F1A]/95 backdrop-blur-sm",
        "transition-[width] duration-200 ease-out",
        open ? "w-56" : "w-14",
      ].join(" ")}
    >
      {/* New Chat */}
      <div className="px-2 pt-3">
        <button
          onClick={click(onNewChat)}
          className={[
            "mx-auto flex items-center justify-center gap-1 rounded-lg px-2.5 py-1.5 text-xs",
            "text-slate-200 hover:bg-white/10 hover:shadow-md hover:shadow-white/10 hover:scale-[1.03]",
            "active:scale-[0.99] transition-all",
            open ? "w-28" : "w-9",
          ].join(" ")}
          title="New chat"
        >
          <span className="text-[14px]" aria-hidden>
            <Icons.New className="h-[14px] w-[14px] text-emerald-300 drop-shadow-[0_0_6px_rgba(16,185,129,0.35)]" />
          </span>
          {open && <span>New Chat</span>}
        </button>
      </div>

      <nav className="mt-2 flex h-[calc(100%-52px)] flex-col gap-1 px-1.5 pb-3">
        {ITEMS.filter((i) => i.key !== "new").map((it) => {
          const isActive = it.key === active;
          const base = "group relative flex items-center gap-2 rounded-lg px-2 py-2 text-sm transition-all";
          const enabled = isActive
            ? "text-slate-100 bg-white/8"
            : "text-slate-300 hover:bg-white/10 hover:shadow-md hover:shadow-white/10 hover:scale-[1.02]";
          const disabled = "text-slate-400 opacity-60 cursor-not-allowed"; // no hover effects

          return (
            <div key={it.key} className="relative">
              <button
                className={`${base} ${it.disabled ? disabled : enabled} w-full text-left`}
                aria-disabled={it.disabled || undefined}
                onClick={it.disabled ? undefined : click(it.onClick)}
                title={it.disabled ? "Soon™" : it.label}
              >
                <span className="text-[16px]">{it.icon}</span>
                {open && <span className="truncate">{it.label}</span>}
              </button>

              {(!open || it.disabled) && (
                <div className="pointer-events-none absolute left-full top-1/2 z-30 ml-2 -translate-y-1/2 whitespace-nowrap rounded-md bg-white/10 px-2 py-1 text-[11px] text-slate-200 opacity-0 shadow transition-opacity group-hover:opacity-100">
                  {it.disabled ? (
                    <>
                      {it.label} <span className="align-super text-[10px]">Soon™</span>
                    </>
                  ) : (
                    it.label
                  )}
                </div>
              )}
            </div>
          );
        })}
      </nav>
    </aside>
  );
}