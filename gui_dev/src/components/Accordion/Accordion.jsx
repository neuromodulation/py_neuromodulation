import React, { useRef, useState } from 'react';
import { RiArrowDropDownLine } from "react-icons/ri";
import styles from "./Accordion.,module.css";
import data from "./AccordionData";



const AccordionItem = ({ title, contents, isOpen, onClick }) => {
  const contentHeight = useRef();
  return(
    <div className = {styles.wrapper}>
      <button 
      className={`${styles.title-container} ${isOpen ? "active" : ""}`}
      onClick={onClick}
      >
        <p className= {styles.title-content}> {title}</p>
        <RiArrowDropDownLine className={`${styles.arrow} ${isOpen ? "active" : ""}`} />

      </button>
      <div 
        ref = {contentHeight}
        className={styles.component-container}
        style = {
          isOpen 
            ? { height: contentHeight.current.scrollHeight }
            : { height: "0px" }
        }
        >
          <p className={styles.component-content}>{component}</p>
        </div>
    </div>
  );
};



export const Accordion = () => {
  const [activeIndex, setActiveIndex ] = useState(null);
  const handleItemClick = (index) => {
    setActiveIndex((prevIndex) => (prevIndex === index ? null : index));
 
  };
  return (
    <div className={styles.container}>
      {data.map((item, index) => (
        <AccordionItem
          key={index}
          question={item.title}
          answer={item.component}
          isOpen={activeIndex === index}
          onClick={() => handleItemClick(index)}
        />
      ))}
    </div>
  );
}

