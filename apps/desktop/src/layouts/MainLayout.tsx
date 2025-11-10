import React, { ReactNode } from "react";
import Header from "../components/Header";
import LeftNav from "../components/LeftNav";
import ModelsRail from "../components/ModelsRail";
import ModelInfoDrawer from "../components/ModelInfoDrawer";
import EmptyState from "../components/EmptyState";
import type { ModelEntry, ModelMeta } from "../types";

type MainLayoutProps = {
  children?: ReactNode;

  // LeftNav
  leftNavActive?: string;
  onOpenModels: () => void;

  // ModelsRail
  modelsRailOpen: boolean;
  models: ModelEntry[];
  selectedModel: ModelEntry | null;
  onSelectModel: (m: ModelEntry) => void;
  onImportModel: () => void | Promise<void>;

  // Info Drawer
  infoOpen: boolean;
  onOpenInfo: () => void;
  onCloseInfo: () => void;
  meta: ModelMeta | null;
  metaLoading: boolean;
  metaError: string | null;

  // Optional: Empty state toggle
  showEmpty?: boolean;
};

export default function MainLayout({
  children,
  leftNavActive = "chat",
  onOpenModels,
  modelsRailOpen,
  models,
  selectedModel,
  onSelectModel,
  onImportModel,
  infoOpen,
  onOpenInfo,
  onCloseInfo,
  meta,
  metaLoading,
  metaError,
  showEmpty = false,
}: MainLayoutProps) {
  const hasChildren = !!children && React.Children.count(children) > 0;
  const renderEmpty = showEmpty || !hasChildren;

  return (
    <div className="relative flex h-screen flex-col bg-[#0B0F1A] text-slate-100 overflow-hidden">
      {/* Header */}
      <Header
        selectedModel={selectedModel}
        recent={models} // using full list as recent for now
        onSelectModel={onSelectModel}
        onOpenInfo={onOpenInfo}
        onImportModel={onImportModel}
      />

      {/* Workspace row (under header) */}
      <div className="relative flex min-h-0 flex-1">
        {/* LeftNav stays on top of anything near it */}
        <div className="relative z-40">
          <LeftNav active={leftNavActive} onOpenModels={onOpenModels} />
        </div>

        {/* Content area */}
        <main className="relative min-w-0 flex-1 min-h-0 overflow-hidden">
          {renderEmpty ? <EmptyState /> : children}
        </main>

        {/* Slide-out ModelsRail: absolutely positioned inside this row.
            - Anchored to the right edge of the nav (left: w-14)
            - Lives UNDER the header because this row starts below header
            - z-30 so nav icons (z-40) remain visible/clickable */}
        <ModelsRail
          open={modelsRailOpen}
          selectedModelId={selectedModel?.id ?? null}
          models={models}
          onPick={onSelectModel}
          onImport={onImportModel}
        />

        {/* Right-side info drawer */}
        <ModelInfoDrawer
          open={infoOpen}
          onClose={onCloseInfo}
          selectedModel={selectedModel}
          meta={meta}
          metaLoading={metaLoading}
          metaError={metaError}
        />
      </div>
    </div>
  );
}