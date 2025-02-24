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

const generateIdFromTitle = (title) => {
  return title.toLowerCase().replace(/\s+/g, "-");
};

export const CollapsibleBox = ({
  id,
  title = "Collapsible Box",
  defaultExpanded = true,
  children,
  headerProps,
  contentProps,
  isolated = false,
  ...props
}) => {
  const boxId = id || generateIdFromTitle(title);

  const toggleAccordionState = useUiStore(
    (state) => state.toggleAccordionState
  );
  const initAccordionState = useUiStore((state) => state.initAccordionState);

  const isExpanded = useUiStore((state) => state.accordionStates[boxId]);

  useEffect(() => {
    initAccordionState(boxId, defaultExpanded);
  }, [boxId, defaultExpanded, initAccordionState]);

  const handleChange = () => {
    toggleAccordionState(boxId);
  };

  // Ensure expanded prop is always boolean (can't never be undefined or MUI will complain)
  const expandedState = isExpanded === undefined ? defaultExpanded : isExpanded;

  const result = (
    <Accordion
      expanded={expandedState}
      onChange={handleChange}
      disableGutters
      {...props}
      sx={{
        overflow: "hidden",
        ...(props?.sx || {}),
      }}
    >
      <AccordionSummary
        expandIcon={<ExpandMoreIcon />}
        id={`${boxId}-header`}
        {...headerProps}
        sx={{
          ...(headerProps?.sx || {}),
        }}
      >
        <Typography variant="h6">{title}</Typography>
      </AccordionSummary>
      <AccordionDetails
        {...contentProps}
        sx={{
          bgcolor: "background.level2",
          ...(contentProps?.sx || {}),
        }}
      >
        {children}
      </AccordionDetails>
    </Accordion>
  );

  return isolated ? <Box>{result}</Box> : result;
};
