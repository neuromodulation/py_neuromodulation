// https://github.com/welldone-software/why-did-you-render/issues/243#issuecomment-1181045461

import * as React from "react";

if (
  import.meta.env.DEV &&
  import.meta.env.VITE_ENABLE_WHY_DID_YOU_RENDER === "true"
) {
  try {
    const { default: wdyr } = await import(
      "@welldone-software/why-did-you-render"
    );

    wdyr(React, {
      include: [/.*/],
      exclude: [/^BrowserRouter/, /^Link/, /^Route/],
      trackHooks: true,
      trackAllPureComponents: true,
    });
  } catch {
    console.error("'Why Did You Render' not found");
  }
}

if (import.meta.env.DEV && import.meta.env.VITE_ENABLE_REACT_SCAN === "true") {
  try {
    const { scan } = await import("react-scan");
    scan({
      enabled: true,
      log: true, // logs render info to console
    });
  } catch {
    console.error("'React Scan' not found");
  }
}
