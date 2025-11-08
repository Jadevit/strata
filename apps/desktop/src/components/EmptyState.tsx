import React from "react";

export default function EmptyState() {
  return (
    <div className="flex h-full w-full items-center justify-center">
      <div className="flex flex-col items-center text-center">
        {/* glassy orb */}
        <div className="relative mb-6">
          <div className="absolute inset-0 rounded-full bg-[radial-gradient(ellipse_at_center,rgba(100,130,255,0.35),rgba(255,255,255,0)_70%)] blur-xl" />
          <div className="relative h-20 w-20 rounded-full border border-white/10 bg-white/5 backdrop-blur-md flex items-center justify-center">
            <span className="text-3xl">🧠</span>
          </div>
        </div>

        {/* headline */}
        <h1 className="text-xl font-semibold text-slate-200">Welcome to Strata</h1>
        <p className="mt-2 text-[15px] text-slate-400 max-w-sm">
          Start by opening a chat, browsing your models, or exploring plugins in the store.
        </p>

        {/* quick action row */}
        <div className="mt-6 flex flex-wrap justify-center gap-3">
          <button
            className="rounded-xl bg-white/10 px-4 py-2 text-sm font-medium text-slate-200 hover:bg-white/15 active:bg-white/20 backdrop-blur-sm border border-white/10 transition"
            onClick={() => console.log('new chat')}
          >
            + New Chat
          </button>
          <button
            className="rounded-xl bg-[#5E73FF]/90 px-4 py-2 text-sm font-medium text-white hover:bg-[#5369ff] active:bg-[#465dff] transition"
            onClick={() => console.log('open models')}
          >
            Browse Models
          </button>
          <button
            className="rounded-xl bg-white/10 px-4 py-2 text-sm font-medium text-slate-200 hover:bg-white/15 active:bg-white/20 backdrop-blur-sm border border-white/10 transition"
            onClick={() => console.log('open store')}
          >
            Open Plugin Store
          </button>
        </div>
      </div>
    </div>
  );
}