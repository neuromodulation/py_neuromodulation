import { create } from "zustand";
import { immer } from "zustand/middleware/immer";
import { devtools, persist as persistMiddleware } from "zustand/middleware";

export const createStore = (
  name,
  initializer,
  persist = false,
  dev = false
) => {
  const fn = persist
    ? persistMiddleware(immer(initializer), name)
    : immer(initializer);

  const dev_fn = dev
    ? devtools(fn, {
        name: name,
      })
    : fn;

  return create(dev_fn);
};

export const createPersistStore = (name, initializer) => {
  return createStore(name, initializer, true);
};
