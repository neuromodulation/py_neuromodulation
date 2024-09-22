import { useEffect, useRef, useState } from "react";
import { useWebviewStore } from "@/stores";

const ResizeIcon = (props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width={800}
    height={800}
    fill="currentColor"
    viewBox="0 0 24 24"
    {...props}
  >
    <path
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="m21 15-6 6m6-13L8 21"
    />
  </svg>
);

export const ResizeHandle = () => {
  const isWebviewReady = useWebviewStore((state) => state.isWebviewReady);
  const resizeHandleRef = useRef(null);
  const [isResizing, setIsResizing] = useState(false);
  const [resizeStyle, setResizeStyle] = useState({});

  // const debouncedResize = useRef(
  //   debounce((width, height) => {
  //     if (isWebviewReady && window.pywebview && window.pywebview.api) {
  //       window.pywebview.api.set_size(width, height);
  //     }
  //   }, 100)
  // ).current;

  useEffect(() => {
    let startX, startY, startWidth, startHeight;

    const handleMouseDown = (event) => {
      setIsResizing(true);
      startX = event.clientX;
      startY = event.clientY;
      startWidth = window.innerWidth;
      startHeight = window.innerHeight;
    };

    const handleMouseMove = (event) => {
      if (!isResizing) return;

      const newWidth = startWidth + event.clientX - startX;
      const newHeight = startHeight + event.clientY - startY;

      setResizeStyle({
        width: `${newWidth}px`,
        height: `${newHeight}px`,
      });

      // debouncedResize(newWidth, newHeight);
    };

    const handleMouseUp = () => {
      setIsResizing(false);
      setResizeStyle({});
    };

    const resizeHandle = resizeHandleRef.current;

    if (resizeHandle) {
      resizeHandle.addEventListener("mousedown", handleMouseDown);
      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleMouseUp);
    }

    return () => {
      if (resizeHandle) {
        resizeHandle.removeEventListener("mousedown", handleMouseDown);
        window.removeEventListener("mousemove", handleMouseMove);
        window.removeEventListener("mouseup", handleMouseUp);
      }
    };
  }, [isWebviewReady, isResizing]);

  useEffect(() => {
    if (isResizing) {
      document.body.style.pointerEvents = "none";
      document.body.style.userSelect = "none";
    } else {
      document.body.style.pointerEvents = "";
      document.body.style.userSelect = "";
    }
  }, [isResizing]);

  return (
    <div ref={resizeHandleRef}>
      <ResizeIcon />
    </div>
  );
};
