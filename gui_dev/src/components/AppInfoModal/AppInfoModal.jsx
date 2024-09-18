import { Button } from "@mui/material";
import { useAppInfoStore } from "@/stores/appInfoStore";
import { SafeExternalLink as ExtLink } from "@/components/utils";
import styles from "./AppInfoModal.module.css";

// Auxiliary component defined outside
const InfoItem = ({ label, value, isLink = false }) => {
  const isEmpty =
    value == null ||
    value === "" ||
    (Array.isArray(value) && value.length === 0);

  if (isEmpty) {
    return null; // Don't render anything if the value is empty
  }

  return (
    <p>
      <span className={styles.infoLabel}>{label}:</span>{" "}
      {isLink ? (
        <ExtLink href={value}>{value}</ExtLink>
      ) : (
        <span className={styles.infoValue}>{value}</span>
      )}
    </p>
  );
};

export const AppInfoModal = ({ onClose }) => {
  const appInfo = useAppInfoStore();

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modalContent} onClick={(e) => e.stopPropagation()}>
        <h2>About PyNeuromodulation</h2>
        <div className={styles.modalItems}>
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

          <Button variant="contained" onClick={onClose}>
            Close
          </Button>
        </div>
      </div>
    </div>
  );
};
