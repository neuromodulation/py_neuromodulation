import { useWebviewStore } from "@/stores";

export const WebviewStatus = () => {
  const webviewStatus = useWebviewStore((state) => state.statusMessage);

  return <span>{webviewStatus}</span>;
};
