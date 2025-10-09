import React from "react";
import type { ModelEntry, ModelMeta } from "../types";

function InfoDot({ colorClass = "bg-green-500" }: { colorClass?: string }) {
  return <span className={`inline-block h-2 w-2 rounded-full ${colorClass}`} aria-hidden="true" />;
}

function InfoI({ title }: { title: string }) {
  return (
    <span
      className="ml-2 inline-flex h-4 w-4 items-center justify-center rounded-full bg-white/10 text-[10px] text-slate-300"
      title={title}
      aria-label="info"
    >
      i
    </span>
  );
}

export default function ModelInfoDrawer({
  open,
  onClose,
  selectedModel,
  meta,
  metaLoading,
  metaError,
}: {
  open: boolean;
  onClose: () => void;
  selectedModel: ModelEntry | null;
  meta: ModelMeta | null;
  metaLoading: boolean;
  metaError: string | null;
}) {
  const status = (() => {
    if (metaLoading) return { text: "Loading…", color: "bg-amber-500" };
    if (metaError)  return { text: "Unavailable", color: "bg-rose-500" };
    if (meta)       return { text: "Ready", color: "bg-emerald-500" };
    return { text: "—", color: "bg-slate-500" };
  })();

  const quant = meta?.quantization ?? "—";
  const ctxWin = meta?.context_length ? `${meta.context_length.toLocaleString()} tokens` : "—";
  const vocab = meta?.vocab_size ? meta.vocab_size.toLocaleString() : "—";
  const eosBos =
    meta?.eos_token_id != null || meta?.bos_token_id != null
      ? `${meta?.eos_token_id ?? "—"} / ${meta?.bos_token_id ?? "—"}`
      : "—";

  const tokenizer =
    meta?.raw?.["tokenizer.ggml.model"] ||
    meta?.raw?.["tokenizer.model"] ||
    meta?.raw?.["tokenizer.type"] ||
    "—";

  const promptFlavor =
    meta?.has_chat_template
      ? "Native template"
      : meta?.prompt_flavor_hint ?? "—";

  const chatTemplate = meta?.has_chat_template ? "Yes" : "No";

  return (
    <>
      {/* scrim */}
      <div
        className={[
          "fixed inset-0 bg-black/40 transition-opacity",
          open ? "opacity-100 pointer-events-auto z-40" : "opacity-0 pointer-events-none z-[-1]",
        ].join(" ")}
        onClick={onClose}
        aria-hidden
      />
      {/* panel */}
      <aside
        className={[
          "fixed right-0 top-0 z-50 h-full w-[400px] bg-[#0F1627] text-slate-100",
          "flex flex-col",
          "transition-transform duration-200",
          open ? "translate-x-0" : "translate-x-full",
        ].join(" ")}
        role="dialog"
        aria-label="Model information"
      >
        {/* sticky header inside panel */}
        <div className="shrink-0 sticky top-0 z-10 flex items-center justify-between border-b border-white/10 px-4 py-3 bg-[#0F1627]">
          <div className="text-sm font-semibold">Model Info</div>
          <button
            className="rounded-md px-2 py-1 text-slate-300 hover:bg-white/5"
            onClick={onClose}
            aria-label="Close model info"
          >
            ✕
          </button>
        </div>

        {/* scrollable content area */}
        <div className="min-h-0 flex-1 overflow-y-auto p-4">
          {/* First row: Name + Status */}
          <div className="mb-3 flex items-center justify-between gap-3">
            <div className="min-w-0">
              <div className="text-[13px] text-slate-400">Name</div>
              <div className="truncate text-sm font-medium text-slate-100">
                {meta?.name ?? selectedModel?.name ?? "—"}
              </div>
            </div>
            <div className="flex shrink-0 items-center gap-2 rounded-full bg-white/5 px-2.5 py-1 text-xs text-slate-200">
              <InfoDot colorClass={status.color} />
              <span className="select-none">{status.text}</span>
            </div>
          </div>

          {/* Two-column key data */}
          <div className="grid grid-cols-2 gap-3">
            <KV label="Backend" value={meta?.backend ?? selectedModel?.backend_hint ?? "—"} />
            <KV label="File Type" value={(meta?.file_type ?? selectedModel?.file_type)?.toUpperCase()} />
            <KV
              label={
                <>
                  Quantization <InfoI title="Compression scheme that trades quality for speed/memory." />
                </>
              }
              value={quant}
            />
            <KV
              label={
                <>
                  Context Window <InfoI title="Max tokens the model can process at once (prompt + response)." />
                </>
              }
              value={ctxWin}
            />
          </div>

          {/* Divider */}
          <div className="my-4 h-px bg-white/10" />

          {/* Advanced */}
          <details className="group" open>
            <summary className="cursor-pointer select-none text-[13px] font-semibold text-slate-200 hover:text-white">
              Advanced
            </summary>
            <div className="mt-3 grid grid-cols-2 gap-3">
              <KV label="Tokenizer" value={tokenizer} />
              <KV label="Vocab Size" value={vocab} />
              <KV label="Prompt Flavor" value={promptFlavor} />
              <KV label="Chat Template" value={chatTemplate} />
              <KV label="EOS / BOS" value={eosBos} />
              <KV label="Quant Label" value={quant} />
              <KV
                label="Path"
                value={<span className="font-mono text-[12px] text-slate-300">{selectedModel?.path ?? "—"}</span>}
              />
            </div>
          </details>

          {metaError && (
            <div className="mt-3 rounded-md bg-rose-500/10 px-3 py-2 text-xs text-rose-300">
              {metaError}
            </div>
          )}
        </div>
      </aside>
    </>
  );
}

function KV({
  label,
  value,
}: {
  label: React.ReactNode;
  value?: React.ReactNode;
}) {
  return (
    <div className="min-w-0">
      <div className="text-[12px] text-slate-400">{label}</div>
      <div className="truncate text-sm text-slate-100/90">{value ?? "—"}</div>
    </div>
  );
}
