import styles from "./Dashboard.module.css";

import {
  Settings,
  Graph,
  Sidebar,
  SidebarDrawer,
  CollapsibleBox,
} from "@/components";

export const Dashboard = () => (
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
