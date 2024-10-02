import { Container } from "@mui/material";

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
  <Container
    component="fieldset"
    {...props}
    sx={{
      borderRadius: 5,
      border: "1px solid #555",
      backgroundColor: "#424242",
      padding: 2,
      width: "100%",
      gap: 2,
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      ...(props?.sx || {}),
    }}
  >
    <legend>{title}</legend>
    {children}
  </Container>
);
