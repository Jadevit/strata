import Chat from "./pages/Chat";
import { HwProfileProvider } from "./context/HwProfileContext";

export default function App() {
  return (
    <HwProfileProvider>
      <Chat />
    </HwProfileProvider>
  );
}