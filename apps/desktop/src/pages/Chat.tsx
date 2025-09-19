import { useEffect, useRef, useState, useCallback } from "react";
import Header from "../components/Header";
import LeftNav from "../components/LeftNav";
import ModelsRail from "../components/ModelsRail"; // <-- slide-in panel
import ChatTranscript from "../components/ChatTranscript";
import Composer from "../components/Composer";
import ModelInfoDrawer from "../components/ModelInfoDrawer";
import { useModels } from "../hooks/useModels";
import { useModelMeta } from "../hooks/useModelMeta";
import { useLLM } from "../hooks/useLLM";

export default function Chat() {
  const [infoOpen, setInfoOpen] = useState(false);
  const [navOpen, setNavOpen] = useState(true);

  // slide-in models rail mode (replaces left nav content, but keeps rail expanded)
  const [modelsMode, setModelsMode] = useState(false);

  // models
  const { models, selectedModel, setSelectedModel, importFromDialog, recent } = useModels();

  // metadata
  const { meta, metaLoading, metaError } = useModelMeta(selectedModel);

  // llm
  const {
    messages,
    input,
    setInput,
    isGenerating,
    setStreamingEnabled,
    sendMessage,
    stop,
    newChat,
  } = useLLM(selectedModel?.id);

  // autoscroll on new messages
  const chatEndRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ensure rail is open before switching to models
  const openModelsRail = useCallback(() => {
    if (!navOpen) {
      setNavOpen(true);
      // toggle mode on the next frame to avoid the “narrow panel” flash
      requestAnimationFrame(() => setModelsMode(true));
    } else {
      setModelsMode(true);
    }
  }, [navOpen]);

  // when a model is picked from the rail, return to normal nav (keep expanded)
  const handlePickModel = useCallback(
    (m: typeof models[number]) => {
      setSelectedModel(m);
      setModelsMode(false); // slide back to nav, do NOT collapse
    },
    [setSelectedModel]
  );

  // hotkeys
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.ctrlKey && (e.key === "i" || e.key === "I")) {
        e.preventDefault();
        setInfoOpen((v) => !v);
      }
      if (e.ctrlKey && (e.key === "b" || e.key === "B")) {
        e.preventDefault();
        setNavOpen((v) => !v);
      }
      if (e.key === "Escape") {
        setInfoOpen(false);
        if (isGenerating) stop();
      }
      // TEMP streaming toggle until settings UI
      if (e.ctrlKey && (e.key === "t" || e.key === "T")) {
        e.preventDefault();
        setStreamingEnabled((v) => !v);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [isGenerating, stop, setStreamingEnabled]);

  return (
    <div className="fixed inset-0 flex flex-col bg-[#0B0F1A] text-slate-100">
      <Header
        selectedModel={selectedModel}
        models={models}
        onSelectModel={(m) => setSelectedModel(m)}
        onOpenInfo={() => setInfoOpen(true)}
        onToggleNav={() => setNavOpen((v) => !v)}
        onImportModel={importFromDialog}
        recent={recent}
      />

      <div className="flex flex-1 min-h-0">
        {/* Left rail: either the normal nav OR the models panel, but never collapse on selection */}
        {modelsMode ? (
          <ModelsRail
            open={navOpen}
            selectedModelId={selectedModel?.id ?? null}
            models={models}
            onBack={() => setModelsMode(false)}
            onPick={handlePickModel}
            onImport={importFromDialog}
          />
        ) : (
          <LeftNav
            open={navOpen}
            active="chat"
            onNewChat={() => newChat()}
            onOpenModels={openModelsRail}
            // If the rail is collapsed and any nav item is clicked, expand first
            onRequestOpen={() => setNavOpen(true)}
          />
        )}

        <div className="flex min-w-0 flex-1 min-h-0 flex-col">
          <ChatTranscript messages={messages} endRef={chatEndRef} />
          <Composer
            value={input}
            disabled={isGenerating}
            isGenerating={isGenerating}
            onChange={(v) => setInput(v)}
            onSend={() => {
              void sendMessage();
            }}
            onCancel={() => {
              void stop();
            }}
          />
        </div>
      </div>

      <ModelInfoDrawer
        open={infoOpen}
        onClose={() => setInfoOpen(false)}
        selectedModel={selectedModel}
        meta={meta}
        metaLoading={metaLoading}
        metaError={metaError}
      />
    </div>
  );
}