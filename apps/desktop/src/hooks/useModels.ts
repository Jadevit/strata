import { useEffect, useState, useCallback, useMemo } from "react";
import type { ModelEntry } from "../types";
import {
  getModelList,
  getActiveModel,
  setActiveModelCmd,
  importModel,
} from "../lib/api";
import { open } from "@tauri-apps/plugin-dialog";

const RECENT_KEY = "strata.recentModels.v1";
const MAX_RECENT = 5;

function loadRecentIds(): string[] {
  try {
    const raw = localStorage.getItem(RECENT_KEY);
    if (!raw) return [];
    const arr = JSON.parse(raw);
    return Array.isArray(arr) ? (arr.filter((x) => typeof x === "string") as string[]) : [];
  } catch {
    return [];
  }
}

function saveRecentIds(ids: string[]) {
  try {
    localStorage.setItem(RECENT_KEY, JSON.stringify(ids.slice(0, MAX_RECENT)));
  } catch {
    /* ignore */
  }
}

export function useModels() {
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [selectedModel, setSelectedModel] = useState<ModelEntry | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [recentIds, setRecentIds] = useState<string[]>(() => loadRecentIds());

  const recent = useMemo(() => {
    if (!models.length || !recentIds.length) return [];
    const byId = new Map(models.map((m) => [m.id, m]));
    const ordered = recentIds.map((id) => byId.get(id)).filter(Boolean) as ModelEntry[];
    // de-dupe (just in case) and cap to MAX_RECENT
    const seen = new Set<string>();
    const unique = ordered.filter((m) => (seen.has(m.id) ? false : (seen.add(m.id), true)));
    return unique.slice(0, MAX_RECENT);
  }, [models, recentIds]);

  const recordRecent = useCallback((id: string) => {
    setRecentIds((prev) => {
      const next = [id, ...prev.filter((x) => x !== id)];
      saveRecentIds(next);
      return next;
    });
  }, []);

  const select = useCallback(async (m: ModelEntry) => {
    setSelectedModel(m);
    recordRecent(m.id);
    try {
      await setActiveModelCmd(m.id);
    } catch (e) {
      console.error("set_active_model_cmd failed:", e);
    }
  }, [recordRecent]);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const list = await getModelList();
      setModels(list);

      if (list.length === 0) {
        setSelectedModel(null);
        return;
      }

      const activeId = await getActiveModel();
      const byId = new Map(list.map((m) => [m.id, m]));

      // Choose selection:
      // 1) active (if present)
      // 2) most recent from MRU that exists
      // 3) fallback to first in list
      let pick: ModelEntry | undefined;
      if (activeId && byId.has(activeId)) {
        pick = byId.get(activeId);
      } else {
        const firstRecent = loadRecentIds().find((id) => byId.has(id));
        pick = (firstRecent && byId.get(firstRecent)) || list[0];
      }
      if (pick) await select(pick);
    } catch (e) {
      setError(String(e));
      setModels([]);
      setSelectedModel(null);
    } finally {
      setLoading(false);
    }
  }, [select]);

  // Import from file picker (plugin-dialog)
  const importFromDialog = useCallback(async () => {
    // Allow multi-select; filter known model extensions
    const picked = await open({
      multiple: true,
      directory: false,
      filters: [{ name: "Models", extensions: ["gguf", "bin", "safetensors", "onnx"] }],
    });

    if (!picked) return;

    const files = Array.isArray(picked) ? picked : [picked];

    // Import sequentially (keeps it simple); collect the last imported to select
    let lastImported: string | null = null;
    for (const srcPath of files) {
      try {
        const entry = await importModel(srcPath);
        lastImported = entry.id;
      } catch (e) {
        console.error("import_model failed for", srcPath, ":", e);
      }
    }

    await refresh();

    if (lastImported) {
      const entry = (models.length ? models : await getModelList()).find(m => m.id === lastImported);
      if (entry) await select(entry);
    }
  }, [refresh, select, models]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return {
    models,
    selectedModel,
    setSelectedModel: select,
    refresh,
    loading,
    error,
    recent,
    importFromDialog,
  };
}