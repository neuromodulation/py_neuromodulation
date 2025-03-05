import { useSessionStore } from "@/stores";
import { useNavigate } from "react-router-dom";

import {
  Box,
  Button,
  Container,
  MenuItem,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  TextField,
  Select,
  Stack,
  Switch,
  Tooltip,
} from "@mui/material";

import { LinkButton } from "@/components/utils";

/**
 * Possible channel types for LSL streams
 */
const channelTypes = Object.freeze([
  "eeg",
  "meg (mag)",
  "meg (grad)",
  "ecg",
  "seeg",
  "dbs",
  "ecog",
  "fnirs (hbo)",
  "fnirs (hbr)",
  "emg",
  "bio",
  "stim",
  "resp",
  "chpi",
  "exci",
  "ias",
  "syst",
]);

/**
 * Possible status options for channels
 */
const statusOptions = Object.freeze(["good", "bad"]);

/**
 * Wrapper for the TextField component
 * @param {object} props - Props for the inner TextField component
 * @returns {JSX.Element} - The TextField component
 */
const TableTextField = (props) => <TextField variant="standard" {...props} />;

/**
 * Wrapper for the Select component
 * @param {object} props - Props for the inner Select component
 * @param {Array} props.options - Array of options for the Select component
 * @returns {JSX.Element} - The Select component
 */
const TableSelect = ({ options, ...props }) => (
  <Select variant="standard" {...props}>
    {options.map((opt, idx) => (
      <MenuItem key={idx} value={opt}>
        {opt}
      </MenuItem>
    ))}
  </Select>
);

/**
 * Array mapping column names to their corresponding components and props
 */
const columns = [
  {
    id: "name",
    label: "Name",
    component: TableTextField,
    description: "Channel name",
  },
  {
    id: "rereference",
    label: "Rereference",
    component: TableTextField,
    description: "Channel to re-reference against",
  },
  {
    id: "type",
    label: "Type",
    component: TableSelect,
    props: { options: channelTypes },
    description: "Channel type",
  },
  {
    id: "status",
    label: "Status",
    component: TableSelect,
    props: { options: statusOptions },
  },
  {
    id: "used",
    label: "Used",
    component: Switch,
  },
  {
    id: "target",
    label: "Target",
    component: Switch,
  },
  {
    id: "new_name",
    label: "New Name",
    component: TableTextField,
  },
];

/**
 *
 * @returns {JSX.Element} - The ChannelsTable component
 */
const ChannelsTable = () => {
  const channels = useSessionStore((state) => state.channels);
  const updateChannel = useSessionStore((state) => state.updateChannel);

  const handleChange = (event, index, field) => {
    let value;
    if (event.target.type === "checkbox") {
      value = event.target.checked ? 1 : 0;
    } else {
      value = event.target.value;
    }
    updateChannel(index, field, value);
  };

  return (
    <TableContainer
      component={Paper}
      sx={{ maxHeight: "80vh", overflowY: "auto" }}
    >
      <Table stickyHeader>
        <TableHead>
          <TableRow>
            {columns.map((column) => (
              <Tooltip
                key={column.id}
                title={column.description}
                placement="top"
              >
                <TableCell
                  key={column.id}
                  align="center"
                  sx={{ p: 1, backgroundColor: "primary.main" }}
                >
                  {column.label}
                </TableCell>
              </Tooltip>
            ))}
          </TableRow>
        </TableHead>
        <TableBody>
          {channels &&
            channels.length > 0 &&
            channels.map((channel, index) => (
              <TableRow
                key={index}
                sx={{
                  backgroundColor: index % 2 ? "background.main" : "#666666",
                }}
              >
                {columns.map((column) => {
                  const Component = column.component;

                  if (Component === Switch) {
                    const switchProps = {
                      onChange: (e) => handleChange(e, index, column.id),
                      checked: channel[column.id] === 1,
                      ...column.props,
                    };
                    return (
                      <TableCell key={column.id} sx={{ py: 1 }}>
                        <Component {...switchProps} />
                      </TableCell>
                    );
                  } else {
                    const otherProps = {
                      onChange: (e) => handleChange(e, index, column.id),
                      value: channel[column.id],
                      ...column.props,
                    };
                    return (
                      <TableCell key={column.id} sx={{ py: 1 }}>
                        <Component {...otherProps} />
                      </TableCell>
                    );
                  }
                })}
              </TableRow>
            ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

/**
 * Channels page component
 * @returns {JSX.Element} - The Channels component
 */
export const Channels = () => {
  const navigate = useNavigate();
  const uploadChannels = useSessionStore((state) => state.uploadChannels);

  return (
    <Container>
      <Stack sx={{ pt: 2 }}>
        <ChannelsTable />
        <Box
          sx={{
            marginTop: 3,
            display: "flex",
            gap: 2,
            justifyContent: "center",
          }}
        >
          <Button
            variant="contained"
            color="primary"
            onClick={() => {
              uploadChannels();
              navigate('/settings');
            }}
          >
            Settings
          </Button>
        </Box>
      </Stack>
    </Container>
  );
};
