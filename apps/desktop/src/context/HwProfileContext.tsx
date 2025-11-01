import { createContext, useContext, type ReactNode } from "react";
import { useHwProfile } from "../hooks/useHwProfile";
import type { HardwareProfile } from "../types";

const HwProfileContext = createContext<HardwareProfile | null>(null);

export function HwProfileProvider({ children }: { children: ReactNode }) {
  const profile = useHwProfile(); // subscribes to the Tauri event
  return (
    <HwProfileContext.Provider value={profile}>
      {children}
    </HwProfileContext.Provider>
  );
}

export function useHwProfileCtx() {
  return useContext(HwProfileContext); // returns HardwareProfile | null
}