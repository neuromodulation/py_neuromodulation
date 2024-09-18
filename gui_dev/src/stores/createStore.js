import { create } from "zustand";
import { immer } from "zustand/middleware/immer";
import { devtools, persist as persistMiddleware } from "zustand/middleware";

export const createStore = (name, initializer, persist = false) => {
  const fn = persist
    ? persistMiddleware(immer(initializer), name)
    : immer(initializer);

  return create(
    devtools(fn, {
      name: name,
    })
  );
};

export const createPersistStore = (name, initializer) => {
  return createStore(name, initializer, true);
};
