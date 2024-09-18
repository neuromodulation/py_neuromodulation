import { Box } from "@mui/material";

/**
 * Component that uses the Box component to render an HTML fieldset element
 * with a legend, and a background with a rounded border. *
 * @param {object} props - Props for the inner Box component
 * @param {string} props.title - The title of the fieldset
 * @param {React.ReactNode} props.children - The content of the fieldset
 * @returns {JSX.Element} The rendered fieldset element
 */
export const TitledBox = ({
  title = "REMEMBER TO GIVE ME A TITLE",
  children,
  ...props
}) => (
  <Box
    component="fieldset"
    p={2}
    borderRadius={5}
    border="1px solid #555"
    backgroundColor="#424242"
    {...props}
  >
    <legend>{title}</legend>
    {children}
  </Box>
);
