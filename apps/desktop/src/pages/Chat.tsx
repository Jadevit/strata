import { useEffect, useRef, useState, useCallback } from "react";
import MainLayout from "../layouts/MainLayout";
import ChatTranscript from "../components/ChatTranscript";
import Composer from "../components/Composer";
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
  } = useLLM(selectedModel?.id);

  // autoscroll on new messages
  const chatEndRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const toggleModelsRail = useCallback(() => {
    setModelsOpen((v) => !v);
  }, []);

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
    <MainLayout
      // left nav
      leftNavActive="chat"
      onOpenModels={toggleModelsRail}

      // models rail
      modelsRailOpen={modelsOpen}
      models={models}
      selectedModel={selectedModel}
      onSelectModel={handlePickModel}
      onImportModel={importFromDialog}

      // info drawer
      infoOpen={infoOpen}
      onOpenInfo={() => setInfoOpen(true)}
      onCloseInfo={() => setInfoOpen(false)}
      meta={meta}
      metaLoading={!!metaLoading}
      metaError={metaError}

      // optional: EmptyState handled by MainLayout if children are empty; Chat always renders content here
      showEmpty={false}
    >
      <ChatTranscript messages={messages} endRef={chatEndRef} />
      <Composer
        value={input}
        disabled={isGenerating}
        isGenerating={isGenerating}
        onChange={(v) => setInput(v)}
        onSend={() => void sendMessage()}
        onCancel={() => void stop()}
      />
    </MainLayout>
  );
}