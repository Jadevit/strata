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
  active = "chat",
  onOpenModels,
}: {
  active?: string;
  onOpenModels: () => void;
}) {
  const ITEMS: Item[] = [
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
        "w-14 shrink-0 border-r border-white/10 bg-[#0B0F1A]/95 backdrop-blur-sm",
      ].join(" ")}
      aria-label="Primary"
    >
      <nav className="mt-2 flex h-full flex-col gap-1 px-1.5 pb-3">
        {ITEMS.map((it) => {
  const isActive = it.key === active;

  const base =
    "relative grid place-items-center rounded-lg p-2 text-sm transition-all focus:outline-none focus:ring-4 focus:ring-white/10";
  const enabled = isActive
    ? "text-slate-100 bg-white/10"
    : "text-slate-300 hover:bg-white/10 hover:shadow-md hover:shadow-white/10 hover:scale-[1.02]";
  const disabled = "text-slate-400 opacity-60 cursor-not-allowed";

  return (
    <div key={it.key} className="relative">
      {/* make the BUTTON the peer */}
      <button
        className={`${base} ${it.disabled ? disabled : enabled} w-full text-left peer`}
        aria-disabled={it.disabled || undefined}
        aria-current={isActive ? "page" : undefined}
        onClick={it.disabled ? undefined : it.onClick}
        aria-label={it.label}
        title={undefined}
      >
        <span className="text-[16px]" aria-hidden>
          {it.icon}
        </span>
        {isActive && (
          <span
            aria-hidden
            className="pointer-events-none absolute left-0 top-1/2 h-5 w-[2px] -translate-y-1/2 rounded-full bg-sky-300/80"
          />
        )}
      </button>

      {/* sibling extrusion listens to the peer's hover/focus */}
      <div
        className={[
          "pointer-events-none absolute left-full top-1/2 z-30 ml-2 -translate-y-1/2",
          "transition-[width,opacity] duration-150 ease-out",
          "w-0 opacity-0",
          "peer-hover:w-44 peer-hover:opacity-100",
          "peer-focus-visible:w-44 peer-focus-visible:opacity-100",
          "peer-focus:w-44 peer-focus:opacity-100",
        ].join(" ")}
      >
        <div
          className={[
            "flex h-9 items-center overflow-hidden rounded-r-xl border border-white/10 border-l-0 px-3",
"backdrop-blur-[8px]",
"bg-[linear-gradient(to_right,rgba(20,25,38,0.75),rgba(20,25,38,0.45),rgba(255,255,255,0.08))]",
"text-sm font-medium text-slate-100",
"shadow-[inset_0_0_0_1px_rgba(255,255,255,0.05)]",
            isActive ? "shadow-[inset_0_0_0_1px_rgba(255,255,255,0.06)]" : "",
          ].join(" ")}
        >
          <span className="whitespace-nowrap">
            {it.disabled ? (
              <>
                {it.label}{" "}
                <span className="align-super text-[10px] opacity-70">Soon™</span>
              </>
            ) : (
              it.label
            )}
          </span>
        </div>
      </div>
    </div>
  );
})}
      </nav>
    </aside>
  );
}