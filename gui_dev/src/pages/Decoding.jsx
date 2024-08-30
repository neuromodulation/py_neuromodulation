import styles from "./Decoding.module.css";

export const Decoding = () => (
    <div className={styles.dashboardContainer}>
      <Sidebar>
        <SidebarDrawer name="settings">
          <Settings />
        </SidebarDrawer>
        <SidebarDrawer name="another">
          <div>Test</div>
        </SidebarDrawer>
      </Sidebar>
      <div className={styles.dashboard}>
        <Graph />
      </div>
    </div>
  );