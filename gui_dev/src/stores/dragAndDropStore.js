// optionsStore.js
import { create } from "zustand";

export const useOptionsStore = create((set) => ({
  options: [{ id: 1, name: "raw_resampling" }],
  setOptions: (newOptions) => set({ options: newOptions }),
  addOption: (option) =>
    set((state) => {
      if (!state.options.some((opt) => opt.id === option.id)) {
        return { options: [...state.options, option] };
      }
      return {};
    }),
  removeOption: (id) =>
    set((state) => ({
      options: state.options.filter((option) => option.id !== id),
    })),
}));
