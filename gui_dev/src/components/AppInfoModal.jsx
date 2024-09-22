import React from "react";
import {
  Modal,
  Typography,
  Button,
  List,
  ListItem,
  ListItemText,
  Link,
  Stack,
} from "@mui/material";
import { useAppInfoStore } from "@/stores/appInfoStore";
import { SafeExternalLink as ExtLink } from "@/components/utils";

const InfoItem = ({ label, value, isLink = false }) => {
  if (
    value == null ||
    value === "" ||
    (Array.isArray(value) && value.length === 0)
  ) {
    return null;
  }

  return (
    <ListItem>
      <ListItemText disableTypography>
        <Typography display="inline" variant="body1" color="primary.main">
          <strong>{label}: </strong>
        </Typography>
        <Typography display="inline" variant="body1">
          {isLink ? (
            <Link component={ExtLink} href={value} color="text.primary">
              {value}
            </Link>
          ) : (
            value
          )}
        </Typography>
      </ListItemText>
    </ListItem>
  );
};

export const AppInfoModal = ({ open, onClose }) => {
  const appInfo = useAppInfoStore();

  return (
    <Modal
      open={open}
      onClose={onClose}
      sx={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <Stack
        sx={{
          bgcolor: "background.paper",
          width: "fit-content",
          borderRadius: 3,
          border: "2px solid white",
          p: 2,
        }}
      >
        <Typography variant="h5">About PyNeuromodulation</Typography>
        <List>
          <InfoItem label="Version" value={appInfo.version} />
          <InfoItem label="Website" value={appInfo.website} isLink />
          <InfoItem label="Authors" value={appInfo.authors.join(", ")} />
          <InfoItem
            label="Maintainers"
            value={appInfo.maintainers.join(", ")}
          />
          <InfoItem label="Repository" value={appInfo.repository} isLink />
          <InfoItem
            label="Documentation"
            value={appInfo.documentation}
            isLink
          />
          <InfoItem label="License" value={appInfo.license} />
          <InfoItem label="Launch Mode" value={appInfo.launchMode} />
        </List>
        <Button variant="contained" onClick={onClose}>
          Close
        </Button>
      </Stack>
    </Modal>
  );
};
