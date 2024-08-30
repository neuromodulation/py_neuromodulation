import React from "react";
import { Button } from "@mui/material";
import { useAppInfo } from "@/stores/appInfoStore";
import { SafeExternalLink as ExtLink } from "@/components/utils";
import styles from "./AppInfoModal.module.css";

export const AppInfoModal = ({ onClose }) => {
  const appInfo = useAppInfo();

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modalContent} onClick={(e) => e.stopPropagation()}>
        <h2>App Information</h2>
        <div className={styles.modalItems}>
          <p>Version: {appInfo.version}</p>
          <p>
            Website: <ExtLink href={appInfo.website}>{appInfo.website}</ExtLink>
          </p>
          <p>Authors: {appInfo.authors.join(", ")}</p>
          <p>Maintainers: {appInfo.maintainers.join(", ")}</p>
          <p>
            Repository:
            <ExtLink href={appInfo.repository}>{appInfo.repository}</ExtLink>
          </p>
          <p>
            Documentation:
            <ExtLink href={appInfo.documentation}>
              {appInfo.documentation}
            </ExtLink>
          </p>
          <p>License: {appInfo.license}</p>
          <p>Launch Mode: {appInfo.launchMode}</p>
          <Button variant="contained" onClick={onClose}>
            Close
          </Button>
        </div>
      </div>
    </div>
  );
};
