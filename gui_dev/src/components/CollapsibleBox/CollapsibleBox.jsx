import { useEffect } from "react";
import {
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Box,
  Typography,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import { useUiStore } from "@/stores/uiStore";
import styles from "./CollapsibleBox.module.css";

const generateIdFromTitle = (title) => {
  return title.toLowerCase().replace(/\s+/g, "-");
};

export const CollapsibleBox = ({
  id,
  title = "Collapsible Box",
  defaultExpanded = true,
  className,
  children,
  sx,
  headerProps,
  contentProps,
}) => {
  const boxId = id || generateIdFromTitle(title);

  const { toggleAccordionState, initAccordionState } = useUiStore((state) => ({
    toggleAccordionState: state.toggleAccordionState,
    initAccordionState: state.initAccordionState,
  }));
  const isExpanded = useUiStore((state) => state.accordionStates[boxId]);

  useEffect(() => {
    initAccordionState(boxId, defaultExpanded);
  }, [boxId, defaultExpanded, initAccordionState]);

  const handleChange = () => {
    toggleAccordionState(boxId);
  };

  // Ensure expanded prop is always boolean (can't never be undefined or MUI will complain)
  const expandedState = isExpanded === undefined ? defaultExpanded : isExpanded;

  return (
    <Box className={className}>
      <Accordion
        expanded={expandedState}
        onChange={handleChange}
        disableGutters
        square={false}
        sx={sx}
      >
        <AccordionSummary
          expandIcon={<ExpandMoreIcon />}
          aria-controls={`${boxId}-content`}
          id={`${boxId}-header`}
          {...headerProps}
        >
          <Typography variant="h6">{title}</Typography>
        </AccordionSummary>
        <AccordionDetails {...contentProps}>{children}</AccordionDetails>
      </Accordion>
    </Box>
  );
};
