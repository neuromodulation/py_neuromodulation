import React, { useEffect, useRef, useState } from 'react';

import styles from "./CollapsibleBox.module.css";



export const CollapsibleBox = ({ title, startOpen = false, children }) => {
  const contentHeight = useRef(null);
  const [isOpen, setIsOpen] = useState(startOpen);

  useEffect(() => {
    if (contentHeight.current) {
      contentHeight.current.style.height = isOpen
          ? `${contentHeight.current.scrollHeight}px` 
          : "0px";
    }
  }, [isOpen])

  const toggleState = () => setIsOpen(!isOpen);

  return(
    <div className = {styles.wrapper}>
      <div className={`${styles.titleContainer} ${isOpen ? styles.active : ''}`} onClick={toggleState}>
        <p className={`${styles.titleContent} ${isOpen ? styles.active : ''}`}>{title}</p>

      </div>

      <div 
        ref = {contentHeight}
        className={styles.contentContainer}
        style = {{ overflow: 'hidden', transition: 'height 0.3s ease' }}
        >
          {children}
        </div>
    </div>
  );
};






