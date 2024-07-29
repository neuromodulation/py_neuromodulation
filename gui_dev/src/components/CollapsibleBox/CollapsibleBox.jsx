import { useRef, useState, useEffect } from "react";
import styles from "./CollapsibleBox.module.css";

const ArrowIcon = (props) => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="-4.5 0 20 20" {...props}>
    <title>{"arrow_right [#336]"}</title>
    <path
      fillRule="evenodd"
      d="M.366 19.708c.405.39 1.06.39 1.464 0l8.563-8.264a1.95 1.95 0 0 0 0-2.827L1.768.292A1.063 1.063 0 0 0 .314.282a.976.976 0 0 0-.011 1.425l7.894 7.617a.975.975 0 0 1 0 1.414L.366 18.295a.974.974 0 0 0 0 1.413"
    />
  </svg>
);

export const CollapsibleBox = ({ title, startOpen = false, children }) => {
  const collapsingContentRef = useRef(null);
  const [contentHeight, setContentHeight] = useState(0);

  useEffect(() => {
    const updateHeight = () =>
      setContentHeight(collapsingContentRef.current?.scrollHeight ?? 0);

    // Set up ResizeObserver
    const resizeObserver = new ResizeObserver(updateHeight);

    if (collapsingContentRef.current) {
      resizeObserver.observe(collapsingContentRef.current);
    }

    // Clean up
    return () => {
      if (collapsingContentRef.current) {
        resizeObserver.unobserve(collapsingContentRef.current);
      }
    };
  }, []); // Empty dependency array means this effect runs once on mount

  return (
    <div className={styles.wrapper}>
      <div className={styles.header}>
        <div className={styles.arrowContainer}>
          <ArrowIcon className={styles.arrow} />
        </div>
        <h3 className={styles.title}>{title}</h3>
        <input
          type="checkbox"
          className={styles.checkbox}
          defaultChecked={startOpen}
        ></input>
      </div>

      <div
        className={styles.contentWrapper}
        style={{ "--content-height": `${contentHeight}px` }}
      >
        <div className={styles.collapsingContentBox} ref={collapsingContentRef}>
          {children}
        </div>
      </div>
    </div>
  );
};
