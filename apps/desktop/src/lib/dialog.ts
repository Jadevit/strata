import { open } from "@tauri-apps/plugin-dialog";

export async function pickModelFile(): Promise<string | null> {
  const file = await open({
    multiple: false,
    directory: false,
    filters: [
      { name: "Model files", extensions: ["gguf", "safetensors", "onnx", "bin"] },
      { name: "All files", extensions: ["*"] }
    ],
  });
  return typeof file === "string" ? file : null;
}