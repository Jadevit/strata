export default function Composer({
  value,
  disabled,
  isGenerating,
  onChange,
  onSend,
  onCancel,
}: {
  value: string;
  disabled: boolean;
  isGenerating: boolean;
  onChange: (v: string) => void;
  onSend: () => void;
  onCancel: () => void;
}) {
  return (
    <footer className="sticky bottom-0 left-0 right-0 border-t border-white/10 bg-[#0B0F1A] px-4 py-3">
      <div className="mx-auto flex max-w-3xl items-center gap-2">
        <textarea
          className="min-h-[48px] max-h-[160px] flex-1 resize-none rounded-xl border border-white/10 bg-white/5
                     px-4 py-[14px] text-[15px] leading-[20px] text-slate-100 placeholder:text-slate-500 outline-none
                     focus:border-[#8FA2FF]/60 focus:ring-4 focus:ring-[#8FA2FF]/15"
          placeholder={isGenerating ? "Model is generating… (Press Esc to stop)" : "Start chatting..."}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              if (!isGenerating) onSend();
            }
            if (e.key === "Escape" && isGenerating) {
              e.preventDefault();
              onCancel();
            }
          }}
          disabled={disabled}
          rows={1}
        />

        {isGenerating ? (
          <button
            className="shrink-0 rounded-xl bg-rose-600 px-4 py-3 text-sm font-semibold text-white hover:bg-rose-700 active:bg-rose-800
                       disabled:cursor-not-allowed disabled:opacity-60"
            onClick={onCancel}
          >
            Stop
          </button>
        ) : (
          <button
            className="shrink-0 rounded-xl bg-[#5E73FF] px-4 py-3 text-sm font-semibold text-white hover:bg-[#5369ff] active:bg-[#465dff]
                       disabled:cursor-not-allowed disabled:opacity-60"
            onClick={onSend}
            disabled={!value.trim()}
          >
            Send
          </button>
        )}
      </div>
    </footer>
  );
}
