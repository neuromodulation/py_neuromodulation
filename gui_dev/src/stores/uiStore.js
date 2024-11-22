import { createPersistStore } from "./createStore";
import { useEffect } from "react";

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

  // Keep track of which accordions are open throughout the app
  accordionStates: {},
  toggleAccordionState: (id) =>
    set((state) => {
      state.accordionStates[id] = !state.accordionStates[id];
    }),
  initAccordionState: (id, defaultState) =>
    set((state) => {
      if (state.accordionStates[id] === undefined) {
        state.accordionStates[id] = defaultState;
      }
    }),

  // Hook to inject UI elements into the status bar
  statusBarContent: () => {},
  setStatusBarContent: (content) => set({ statusBarContent: content }),
  clearStatusBarContent: () => set({ statusBarContent: null }),
}));

// Use this hook from Page components to inject page-specific UI elements into the status bar
export const useStatusBarContent = (content) => {
  const createStatusBarContent = () => content;

  const setStatusBarContent = useUiStore((state) => state.setStatusBarContent);
  const clearStatusBarContent = useUiStore(
    (state) => state.clearStatusBarContent
  );

  useEffect(() => {
    setStatusBarContent(createStatusBarContent);
    return () => clearStatusBarContent();
  }, [content, setStatusBarContent, clearStatusBarContent]);
};
