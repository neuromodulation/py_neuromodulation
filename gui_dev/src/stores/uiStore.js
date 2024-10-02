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
}));
