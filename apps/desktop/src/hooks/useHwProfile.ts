import { useEffect, useState } from "react";
import { onHwProfile } from "../lib/events";
import type { HardwareProfile } from "../types";

export function useHwProfile() {
  // local state for the detected hardware
  const [profile, setProfile] = useState<HardwareProfile | null>(null);

  useEffect(() => {
    let unlisten: (() => void) | null = null;

    // subscribe to the backend event
    onHwProfile((p) => {
      console.log("[hwprof] received:", p);
      setProfile(p);
    }).then((off) => (unlisten = off));

    // cleanup on component unmount
    return () => {
      if (unlisten) unlisten();
    };
  }, []);

  return profile;
}