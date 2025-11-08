import { useEffect, useRef, useState, useCallback } from "react";
import Header from "../components/Header";
import LeftNav from "../components/LeftNav";
import ModelsRail from "../components/ModelsRail";
import ChatTranscript from "../components/ChatTranscript";
import Composer from "../components/Composer";
import ModelInfoDrawer from "../components/ModelInfoDrawer";
import { useModels } from "../hooks/useModels";
import { useModelMeta } from "../hooks/useModelMeta";
import { useLLM } from "../hooks/useLLM";

export default function Chat() {
  const [infoOpen, setInfoOpen] = useState(false);
  const [modelsOpen, setModelsOpen] = useState(false);

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

  // open models rail
  const openModelsRail = useCallback(() => {
    setModelsOpen((v) => !v);
  }, []);

  // when a model is picked, close the rail
  const handlePickModel = useCallback(
    (m: typeof models[number]) => {
      setSelectedModel(m);
      setModelsOpen(false);
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
      if (e.ctrlKey && (e.key === "m" || e.key === "M")) {
        e.preventDefault();
        setModelsOpen((v) => !v);
      }
      if (e.key === "Escape") {
        setInfoOpen(false);
        setModelsOpen(false);
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
      {/* Header */}
      <Header
        selectedModel={selectedModel}
        recent={recent}
        onSelectModel={setSelectedModel}
        onOpenInfo={() => setInfoOpen(true)}
        onImportModel={importFromDialog}
      />

      <div className="flex flex-1 min-h-0">
        {/* LeftNav (always visible) */}
        <LeftNav active="chat" onOpenModels={openModelsRail} />

        {/* Chat area */}
        <div className="flex min-w-0 flex-1 min-h-0 flex-col relative">
          <ChatTranscript messages={messages} endRef={chatEndRef} />
          <Composer
            value={input}
            disabled={isGenerating}
            isGenerating={isGenerating}
            onChange={(v) => setInput(v)}
            onSend={() => void sendMessage()}
            onCancel={() => void stop()}
          />
        </div>

        {/* Models Rail (slides out beside nav) */}
        <ModelsRail
          open={modelsOpen}
          selectedModelId={selectedModel?.id ?? null}
          models={models}
          onPick={handlePickModel}
          onImport={importFromDialog}
        />

        {/* Model Info Drawer */}
        <ModelInfoDrawer
          open={infoOpen}
          onClose={() => setInfoOpen(false)}
          selectedModel={selectedModel}
          meta={meta}
          metaLoading={!!metaLoading}
          metaError={metaError}
        />
      </div>
    </div>
  );
}