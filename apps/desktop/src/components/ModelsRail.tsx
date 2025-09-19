import { useMemo } from "react";
import type { ModelEntry } from "../types";

type Props = {
  open: boolean;
  selectedModelId: string | null;
  models: ModelEntry[];
  onBack: () => void;
  onPick: (m: ModelEntry) => void;
  onImport: () => void | Promise<void>;
};

function BackendPill({ backend }: { backend: string }) {
  const label = backend === "llama" ? "Llama" : backend.charAt(0).toUpperCase() + backend.slice(1);
  return (
    <span className="ml-2 shrink-0 rounded-full border border-white/15 px-2 py-[2px] text-[10px] text-slate-300">
      {label}
    </span>
  );
}

export default function ModelsRail({
  open,
  selectedModelId,
  models,
  onBack,
  onPick,
  onImport,
}: Props) {
  const groups = useMemo(() => {
    const g = new Map<string, ModelEntry[]>();
    for (const m of models) {
      const key = (m.backend_hint || "unknown").toLowerCase();
      if (!g.has(key)) g.set(key, []);
      g.get(key)!.push(m);
    }
    for (const arr of g.values()) {
      arr.sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: "base" }));
    }
    const keys = Array.from(g.keys()).sort((a, b) => {
      if (a === "llama" && b !== "llama") return -1;
      if (b === "llama" && a !== "llama") return 1;
      return a.localeCompare(b);
    });
    return keys.map((k) => ({ key: k, items: g.get(k)! }));
  }, [models]);

  return (
    <aside
      className={[
        "shrink-0 border-r border-white/10 bg-[#0B0F1A]/95 backdrop-blur-sm",
        "transition-[width] duration-200 ease-out",
        open ? "w-56" : "w-14",
        "flex flex-col",
      ].join(" ")}
    >
      <div className="flex items-center gap-2 px-2 pt-3">
        <button
          onClick={onBack}
          className="grid h-8 w-8 place-items-center rounded-md hover:bg-white/10 focus:outline-none focus:ring-4 focus:ring-white/10 transition"
          title="Back"
          aria-label="Back"
        >
          <span className="text-slate-300 text-[16px]" aria-hidden>←</span>
        </button>
        {open && <div className="text-sm font-semibold text-slate-200">Models</div>}
      </div>

      <div className="mt-2 flex-1 overflow-y-auto px-1.5 pb-3">
        {groups.map(({ key, items }) => {
          const pretty = key === "llama" ? "Llama" : key.charAt(0).toUpperCase() + key.slice(1);
          return (
            <div key={key} className="mb-3">
              {open ? (
                <div className="px-2 py-1 text-[11px] uppercase tracking-wide text-slate-400">
                  {pretty}
                </div>
              ) : (
                <div className="h-2" />
              )}
              <div className="flex flex-col gap-1">
                {items.map((m) => {
                  const supported = key === "llama";
                  const isActive = m.id === selectedModelId;
                  const base = "group relative flex items-center gap-2 rounded-lg px-2 py-2 text-sm";
                  const style = !supported
                    ? "opacity-50 cursor-not-allowed"
                    : isActive
                    ? "bg-white/8 text-slate-100"
                    : "text-slate-300 hover:bg-white/10 hover:shadow-md hover:shadow-white/10 hover:scale-[1.02] transition-all";
                  return (
                    <button
                      key={m.id}
                      className={`${base} ${style}`}
                      onClick={!supported ? undefined : () => onPick(m)}
                      title={!supported ? "Backend not available yet" : m.name}
                      aria-disabled={!supported || undefined}
                    >
                      <span className="text-[16px]" aria-hidden>🧠</span>
                      {open && (
                        <div className="min-w-0 truncate">
                          <span className="truncate">{m.name}</span>
                          <BackendPill backend={m.backend_hint} />
                        </div>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>

      <div className="border-t border-white/10 px-2 py-2">
        <button
          onClick={() => onImport()}
          className="w-full rounded-lg bg-white/5 px-2.5 py-2 text-center text-xs text-slate-200 hover:bg-white/10 transition"
          title="Import Model"
        >
          Import Model
        </button>
      </div>
    </aside>
  );
}