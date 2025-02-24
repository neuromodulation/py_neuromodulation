import styles from "./Sidebar.module.css";
import { Children, useState, useRef, useEffect } from "react";

const DrawerToggle = ({ name, isOpen, onToggle }) => {
  return (
    <button
      className={`${styles.drawerToggle} ${isOpen ? styles.open : ""}`}
      onClick={() => onToggle(name)}
    >
      <span className={styles.toggleLabel}>{name}</span>
    </button>
  );
};

export const SidebarDrawer = ({ name, children, isOpen }) => {
  return (
    <div className={`${styles.drawer} ${isOpen ? styles.open : ""}`}>
      {children}
    </div>
  );
};

export const Sidebar = ({ children }) => {
  const [activeDrawer, setActiveDrawer] = useState(null);
  const [drawerWidth, setDrawerWidth] = useState(300);
  const [lastWidth, setLastWidth] = useState(drawerWidth);
  const [isResizing, setIsResizing] = useState(false);
  const drawerRef = useRef(null);

  const MIN_DRAWER_WIDTH = 50;
  const MAX_DRAWER_WIDTH = 600;

  // Render only the active drawer component
  const drawerChildren = Children.toArray(children).filter(
    (child) => child.type.name === "SidebarDrawer"
  );

  const drawerNames = drawerChildren.map((child) => child.props.name);

  const activeChild = drawerChildren.find(
    (child) => child.props.name === activeDrawer
  );

  // Toggle drawer behavior
  const toggleDrawer = (drawerName) => {
    setActiveDrawer((prev) => {
      if (prev === null) {
        // Opening from closed state
        setDrawerWidth(lastWidth);
      } else if (prev === drawerName) {
        // Closing the drawer
        setLastWidth(drawerWidth);
        return null;
      }
      // Switching between drawers or opening a new one
      return drawerName;
    });
  };

  // Use mouse movement to resize the drawer
  const handleMouseMove = (e) => {
    // Use functional update to make sure we catch the latest isResizing state
    setIsResizing((currentIsResizing) => {
      if (!currentIsResizing) return false;

      // Same with the width
      setDrawerWidth((currentWidth) => {
        const newWidth =
          e.clientX - drawerRef.current.getBoundingClientRect().left;
        const clampedWidth = Math.min(
          Math.max(newWidth, MIN_DRAWER_WIDTH),
          MAX_DRAWER_WIDTH
        );

        // Close drawer if width is very small
        if (clampedWidth <= 50) {
          setActiveDrawer(null);
          stopResizing();
          return 0;
        }

        return clampedWidth;
      });

      return currentIsResizing;
    });
  };

  const startResizing = (e) => {
    setIsResizing(true);
    setLastWidth(drawerWidth);

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", stopResizing);
  };

  const stopResizing = () => {
    setIsResizing(false);
    setLastWidth(drawerWidth);

    document.removeEventListener("mousemove", handleMouseMove);
    document.removeEventListener("mouseup", stopResizing);
  };

  useEffect(() => {
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", stopResizing);
    };
  }, []);

  return (
    <div className={styles.sidebarContainer}>
      <div className={styles.sidebar}>
        {drawerNames.map((name) => (
          <DrawerToggle
            key={name}
            name={name}
            isOpen={activeDrawer === name}
            onToggle={toggleDrawer}
          />
        ))}
      </div>
      <div
        ref={drawerRef}
        className={`${styles.drawerContainer} ${activeDrawer ? styles.open : ""} ${isResizing ? styles.resizing : ""}`}
        style={{ width: activeDrawer ? `${drawerWidth}px` : 0 }}
      >
        {activeChild}
        <div
          className={`${styles.resizeHandle} ${isResizing ? styles.resizing : ""}`}
          onMouseDown={startResizing}
        ></div>
      </div>
    </div>
  );
};
