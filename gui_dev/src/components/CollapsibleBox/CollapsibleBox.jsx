import React, { useRef, useState } from 'react';
import { RiArrowDropDownLine } from "react-icons/ri";
import styles from "./CollapsibleBox.module.css";



export const CollapsibleBox = ({ title, startOpen = False, children }) => {
  const contentHeight = useRef();
  const [isOpen, setIsOpen] = useState(startOpen);
  const toggleState = () => setIsOpen(!isOpen);
  
  
  return(
    <div className = {styles.wrapper}>
      <input
      type = "checkbox"
      checked = {startOpen}
      className={styles.titleContainer}
      >
        <p className= {styles.titleContent}> {title}</p>
        <RiArrowDropDownLine className={styles.arrow}/>

      </input>
      <div 
        ref = {contentHeight}
        className={styles.contentContainer}
        style = {
          isOpen 
            ? { height: contentHeight.current.scrollHeight }
            : { height: "0px" }
        }
        >
          {children}
        </div>
    </div>
  );
};




