import { createPersistStore } from "./createStore";

export const useUiStore = createPersistStore("ui", (set, get) => ({
  activeDrawer: null,
  toggleDrawer: (drawerName) =>
    set((state) => {
      const newActiveDrawer =
        state.activeDrawer === drawerName ? null : drawerName;
      return { activeDrawer: newActiveDrawer };
    }),
  isDrawerOpen: (drawerName) => {
    const isOpen = get().activeDrawer === drawerName;
    return isOpen;
  },
  closeAllDrawers: () => {
    set({ activeDrawer: null });
  },
}));
