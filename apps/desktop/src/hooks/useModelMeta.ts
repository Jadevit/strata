import { useEffect, useState } from "react";
import type { ModelEntry, ModelMeta } from "../types";
import { getModelMetadata } from "../lib/api";

export function useModelMeta(selectedModel: ModelEntry | null) {
  const [meta, setMeta] = useState<ModelMeta | null>(null);
  const [metaLoading, setMetaLoading] = useState(false);
  const [metaError, setMetaError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedModel) {
      setMeta(null);
      setMetaLoading(false);
      setMetaError(null);
      return;
    }
    let cancelled = false;

    setMetaLoading(true);
    setMetaError(null);

    getModelMetadata()
      .then((m) => { if (!cancelled) setMeta(m); })
      .catch((err) => {
        console.error("get_model_metadata failed:", err);
        if (!cancelled) {
          setMeta(null);
          setMetaError(String(err));
        }
      })
      .finally(() => { if (!cancelled) setMetaLoading(false); });

    return () => { cancelled = true; };
  }, [selectedModel]);

  return { meta, metaLoading, metaError };
}