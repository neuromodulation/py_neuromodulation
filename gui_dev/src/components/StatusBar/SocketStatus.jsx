import { useSocketStore } from "@/stores";
import styles from "./StatusBar.module.css";

const DisconnectedIcon = (props) => (
  // Source: https://www.svgrepo.com/svg/384385/delete-remove-uncheck
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width={800}
    height={800}
    viewBox="0 0 60 60"
    {...props}
  >
    <path
      d="M940 510a30 30 0 1 1 30-30 30 30 0 0 1-30 30Zm15-20.047a3.408 3.408 0 0 1 0 4.817l-.221.22a3.42 3.42 0 0 1-4.833 0l-8.764-8.755a1.71 1.71 0 0 0-2.417 0l-8.741 8.747a3.419 3.419 0 0 1-4.836 0l-.194-.193a3.408 3.408 0 0 1 .017-4.842l8.834-8.735a1.7 1.7 0 0 0 0-2.43l-8.831-8.725a3.409 3.409 0 0 1-.018-4.844l.193-.193a3.413 3.413 0 0 1 2.418-1c.944 0 3.255 1.835 3.872 2.455l7.286 7.287a1.708 1.708 0 0 0 2.417 0l8.764-8.748a3.419 3.419 0 0 1 4.832 0l.222.229a3.408 3.408 0 0 1 0 4.818l-8.727 8.737a1.7 1.7 0 0 0 0 2.407Z"
      style={{
        fill: "#9f4c4c",
        fillRule: "evenodd",
      }}
      transform="translate(-910 -450)"
    />
  </svg>
);

const ConnectedIcon = (props) => (
  // Source: https://www.svgrepo.com/svg/384403/accept-check-good-mark-ok-tick
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width={800}
    height={800}
    viewBox="0 0 60 60"
    {...props}
  >
    <path
      d="M800 510a30 30 0 1 1 30-30 30 30 0 0 1-30 30Zm-16.986-23.235a3.484 3.484 0 0 1 0-4.9l1.766-1.756a3.185 3.185 0 0 1 4.574.051l3.12 3.237a1.592 1.592 0 0 0 2.311 0l15.9-16.39a3.187 3.187 0 0 1 4.6-.027l1.715 1.734a3.482 3.482 0 0 1 0 4.846l-21.109 21.451a3.185 3.185 0 0 1-4.552.03Z"
      style={{
        fill: "#699f4c",
        fillRule: "evenodd",
      }}
      transform="translate(-770 -450)"
    />
  </svg>
);

const StatusIcons = {
  connected: ConnectedIcon,
  connecting: DisconnectedIcon,
  disconnected: DisconnectedIcon,
};

export const SocketStatus = () => {
  const socketStatus = useSocketStore((state) => state.status);
  const socketError = useSocketStore((state) => state.error);

  return (
    <span className={styles.socketStatus}>
      SocketIO: {StatusIcons[socketStatus]()}
      {socketError && (
        <>
          {" - "}
          <span className={styles.socketError}> Error: {socketError}</span>
        </>
      )}
    </span>
  );
};
