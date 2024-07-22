import { useUiStore } from "@/stores";
import styles from "./Drawer.module.css";
import { useEffect, useState } from "react";

export const Drawer = ({ name, children }) => {
  const isOpen = useUiStore((state) => state.isDrawerOpen(name));
  const [isVisible, setIsVisible] = useState(isOpen);

  useEffect(() => {
    if (isOpen) {
      setIsVisible(true);
    } else {
      const timer = setTimeout(() => setIsVisible(false), 300); // 300ms matches the CSS transition time
      return () => clearTimeout(timer);
    }
  }, [isOpen]);

  if (!isVisible && !isOpen) return null;

  return <div className={styles.drawer}>{children}</div>;
};
