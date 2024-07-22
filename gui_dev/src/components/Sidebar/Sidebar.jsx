import { useUiStore } from "@/stores";
import styles from "./Sidebar.module.css";

const DrawerToggle = ({ name, label }) => {
  const toggleDrawer = useUiStore((state) => state.toggleDrawer);
  const isOpen = useUiStore((state) => state.isDrawerOpen(name));

  return (
    <button
      className={`${styles.drawerToggle} ${isOpen ? styles.open : ""}`}
      onClick={() => toggleDrawer(name)}
    >
      <span>{label}</span>
    </button>
  );
};

export const Sidebar = ({ children }) => {
  const activeDrawer = useUiStore((state) => state.activeDrawer);

  return (
    <div className={styles.sidebarContainer}>
      <div className={styles.sidebar}>
        <DrawerToggle name="settings" label="Settings" />
      </div>
      <div
        className={`${styles.drawerContainer} ${activeDrawer ? styles.open : ""}`}
      >
        {children}
      </div>
    </div>
  );
};
