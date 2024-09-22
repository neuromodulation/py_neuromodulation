import { useSessionStore } from "@/stores";

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
    updateChannel(
      index,
      field,
      event.target.type === "checkbox"
        ? event.target.checked
        : event.target.value
    );
  };

  return (
    <TableContainer
      component={Paper}
      sx={{ maxHeight: "500px", overflowY: "auto" }}
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
                {columns.map((column) => (
                  <TableCell
                    key={column.id}
                    sx={{
                      py: 1,
                    }}
                  >
                    <column.component
                      value={channel[column.id]}
                      onChange={(e) => handleChange(e, index, column.id)}
                      {...column.props}
                    />
                  </TableCell>
                ))}
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
            onClick={() => uploadChannels()}
          >
            Save Channels
          </Button>
          <LinkButton to="/settings" variant="contained" color="primary">
            Settings
          </LinkButton>
        </Box>
      </Stack>
    </Container>
  );
};
