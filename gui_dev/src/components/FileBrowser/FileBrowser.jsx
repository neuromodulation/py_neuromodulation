import { useReducer, useEffect } from "react";
import {
  Box,
  Paper,
  Typography,
  TextField,
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
  Modal,
  Select,
  FormControl,
  InputLabel,
  Snackbar,
  Menu,
  MenuItem,
  Checkbox,
  ListItemIcon,
  ListItemText,
} from "@mui/material";
import {
  ArrowUpward,
  Folder,
  InsertDriveFile,
  MoreVert,
} from "@mui/icons-material";

import { QuickAccessSidebar } from "./QuickAccess";
import { FileManager } from "@/utils/FileManager";

const fileManager = new FileManager("");

const ALLOWED_EXTENSIONS = [".npy", ".vhdr", ".fif", ".edf", ".bdf"];

const initialState = {
  currentPath: "",
  editablePath: "",
  files: [],
  sortConfig: { key: "name", direction: "asc" },
  drives: [],
  selectedDrive: "",
  error: "",
  showHiddenFiles: false,
  menuAnchorEl: null,
};

function reducer(state, action) {
  switch (action.type) {
    case "SET_CURRENT_PATH":
      return {
        ...state,
        currentPath: action.payload,
        editablePath: action.payload,
      };
    case "SET_EDITABLE_PATH":
      return { ...state, editablePath: action.payload };
    case "SET_FILES":
      return { ...state, files: action.payload };
    case "SET_SORT_CONFIG":
      return { ...state, sortConfig: action.payload };
    case "SET_DRIVES":
      return { ...state, drives: action.payload };
    case "SET_SELECTED_DRIVE":
      return { ...state, selectedDrive: action.payload };
    case "SET_ERROR":
      return { ...state, error: action.payload };
    case "TOGGLE_HIDDEN_FILES":
      return { ...state, showHiddenFiles: !state.showHiddenFiles };
    case "SET_MENU_ANCHOR_EL":
      return { ...state, menuAnchorEl: action.payload };
    default:
      return state;
  }
}

const columns = [
  { key: "name", label: "Name" },
  { key: "type", label: "Type" },
  { key: "size", label: "Size" },
  { key: "created_at", label: "Created" },
  { key: "modified_at", label: "Modified" },
];

export const FileBrowser = ({
  isModal = false,
  directory = null,
  onClose,
  onFileSelect,
}) => {
  const [state, dispatch] = useReducer(reducer, initialState);

  useEffect(() => {
    fetchDrives();
    if (directory) {
      dispatch({ type: "SET_CURRENT_PATH", payload: directory });
    } else {
      fetchHomeDirectory();
    }
  }, []);

  useEffect(() => {
    if (state.currentPath) {
      fetchFiles(state.currentPath);
    }
  }, [state.currentPath, state.showHiddenFiles]);

  const fetchDrives = async () => {
    try {
      const response = await fetch("/api/drives");
      if (!response.ok) throw new Error("Failed to fetch drives");
      const data = await response.json();
      dispatch({ type: "SET_DRIVES", payload: data.drives });
      if (data.drives.length > 0) {
        dispatch({ type: "SET_SELECTED_DRIVE", payload: data.drives[0] });
      }
    } catch (error) {
      console.error("Error fetching drives:", error);
      dispatch({ type: "SET_ERROR", payload: "Failed to fetch drives" });
    }
  };

  const fetchHomeDirectory = async () => {
    try {
      const response = await fetch("/api/home_directory");
      if (!response.ok) throw new Error("Failed to fetch home directory");
      const data = await response.json();
      dispatch({ type: "SET_CURRENT_PATH", payload: data.home_directory });
    } catch (error) {
      console.error("Error fetching home directory:", error);
      dispatch({
        type: "SET_ERROR",
        payload: "Failed to fetch home directory",
      });
    }
  };

  const fetchFiles = async (path) => {
    try {
      const files = await fileManager.getFiles({
        path,
        allowedExtensions: ALLOWED_EXTENSIONS.join(","),
        showHidden: state.showHiddenFiles,
      });
      dispatch({
        type: "SET_FILES",
        payload: files.map((file) => ({
          ...file,
          type: file.is_directory
            ? "Directory"
            : file.name.split(".").pop().toUpperCase(),
        })),
      });
      dispatch({ type: "SET_ERROR", payload: "" });
      return true;
    } catch (error) {
      console.error("Error fetching files:", error);
      dispatch({ type: "SET_ERROR", payload: error.message });
      return false;
    }
  };

  const handleFileClick = (file) => {
    if (file.is_directory) {
      dispatch({ type: "SET_CURRENT_PATH", payload: file.path });
    } else if (
      ALLOWED_EXTENSIONS.some((ext) => file.name.toLowerCase().endsWith(ext))
    ) {
      onFileSelect(file);
    }
  };

  const handleParentDirectory = () => {
    dispatch({
      type: "SET_CURRENT_PATH",
      payload: state.currentPath.split(/[/\\]/).slice(0, -1).join("/") || "/",
    });
  };

  const handleQuickAccessClick = (path) => {
    dispatch({ type: "SET_CURRENT_PATH", payload: path });
  };

  const handleDriveChange = (event) => {
    dispatch({ type: "SET_SELECTED_DRIVE", payload: event.target.value });
    dispatch({ type: "SET_CURRENT_PATH", payload: event.target.value });
  };

  const handlePathChange = (event) => {
    dispatch({ type: "SET_EDITABLE_PATH", payload: event.target.value });
  };

  const handlePathKeyPress = async (event) => {
    if (event.key === "Enter") {
      const isValidPath = await fetchFiles(state.editablePath);
      if (isValidPath) {
        dispatch({ type: "SET_CURRENT_PATH", payload: state.editablePath });
      } else {
        dispatch({ type: "SET_EDITABLE_PATH", payload: state.currentPath });
      }
    }
  };

  const handleSort = (key) => {
    const direction =
      state.sortConfig.key === key && state.sortConfig.direction === "asc"
        ? "desc"
        : "asc";
    dispatch({
      type: "SET_SORT_CONFIG",
      payload: { key, direction },
    });
    const sortedFiles = fileManager.sortFiles(
      [state.files],
      key,
      direction === "asc"
    );
    dispatch({ type: "SET_FILES", payload: sortedFiles });
  };

  const handleMenuOpen = (event) => {
    dispatch({ type: "SET_MENU_ANCHOR_EL", payload: event.currentTarget });
  };

  const handleMenuClose = () => {
    dispatch({ type: "SET_MENU_ANCHOR_EL", payload: null });
  };

  const handleToggleHiddenFiles = () => {
    console.log("Toggle hidden files");
    dispatch({ type: "TOGGLE_HIDDEN_FILES" });
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString();
  };

  const formatSize = (bytes) => {
    const sizes = ["Bytes", "KB", "MB", "GB", "TB"];
    if (bytes === 0) return "0 Byte";
    const i = parseInt(Math.floor(Math.log(bytes) / Math.log(1024)));
    return Math.round(bytes / Math.pow(1024, i), 2) + " " + sizes[i];
  };

  const renderCellContent = (file, column) => {
    switch (column.key) {
      case "name":
        return (
          <Box display="flex" alignItems="center">
            {file.is_directory ? (
              <Folder color="primary" />
            ) : (
              <InsertDriveFile />
            )}
            <Typography component="span" variant="body2" sx={{ ml: 1 }}>
              {file.name}
            </Typography>
          </Box>
        );
      case "size":
        return file.is_directory ? "--" : formatSize(file.size);
      case "created_at":
      case "modified_at":
        return formatDate(file[column.key]);
      default:
        return file[column.key];
    }
  };

  const FileBrowserContent = (
    <Box sx={{ width: "100%", maxWidth: 800, margin: "auto" }}>
      <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
        <Box display="flex" alignItems="center">
          {state.drives.length > 1 && (
            <FormControl sx={{ minWidth: 120, mr: 2 }}>
              <InputLabel id="drive-select-label">Drive</InputLabel>
              <Select
                labelId="drive-select-label"
                id="drive-select"
                value={state.selectedDrive}
                onChange={handleDriveChange}
                size="small"
              >
                {state.drives.map((drive) => (
                  <MenuItem key={drive} value={drive}>
                    {drive}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          )}
          <TextField
            fullWidth
            variant="outlined"
            size="small"
            value={state.editablePath}
            onChange={handlePathChange}
            onKeyPress={handlePathKeyPress}
            error={!!state.error}
            helperText={state.error}
          />
          <IconButton
            onClick={handleParentDirectory}
            disabled={
              state.currentPath === "/" ||
              (state.drives.length > 1 &&
                state.drives.includes(state.currentPath))
            }
            sx={{ ml: 1 }}
          >
            <ArrowUpward />
          </IconButton>
          <IconButton onClick={handleMenuOpen} sx={{ ml: 1 }}>
            <MoreVert />
          </IconButton>
          <Menu
            anchorEl={state.menuAnchorEl}
            open={Boolean(state.menuAnchorEl)}
            onClose={handleMenuClose}
          >
            <MenuItem onClick={handleToggleHiddenFiles}>
              <ListItemIcon>
                <Checkbox
                  checked={state.showHiddenFiles}
                  onChange={handleToggleHiddenFiles}
                  onClick={(e) => e.stopPropagation()} // Prevent MenuItem click
                />
              </ListItemIcon>
              <ListItemText primary="Show Hidden Files" />
            </MenuItem>{" "}
          </Menu>
        </Box>
      </Paper>
      <Box
        sx={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "row",
          maxWidth: 800,
          margin: "auto",
          gap: 2,
        }}
      >
        <QuickAccessSidebar onItemClick={handleQuickAccessClick} />

        <TableContainer
          component={Paper}
          sx={{
            flexBasis: "75%",
            flexShrink: 0,
            maxHeight: 400,
            scrollbarWidth: "thin",
            scrollbarColor: "rgba(0,0,0,.1) transparent",
          }}
        >
          <Table stickyHeader size="small">
            <TableHead>
              <TableRow sx={{ height: "2.5rem" }}>
                {columns.map((column) => (
                  <TableCell key={column.key}>
                    <TableSortLabel
                      active={state.sortConfig.key === column.key}
                      direction={
                        state.sortConfig.key === column.key
                          ? state.sortConfig.direction
                          : "asc"
                      }
                      onClick={() => handleSort(column.key)}
                    >
                      {column.label}
                    </TableSortLabel>
                  </TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {state.files.map((file) => (
                <TableRow
                  key={file.path}
                  hover
                  onClick={() => handleFileClick(file)}
                  sx={{ cursor: "pointer" }}
                >
                  {columns.map((column) => (
                    <TableCell key={column.key}>
                      <Typography component="span" variant="body2">
                        {renderCellContent(file, column)}
                      </Typography>
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Box>
      <Snackbar
        open={!!state.error}
        autoHideDuration={6000}
        onClose={() => dispatch({ type: "SET_ERROR", payload: "" })}
        message={state.error}
      />
    </Box>
  );

  return isModal ? (
    <Modal
      open={true}
      onClose={onClose}
      aria-labelledby="file-browser-modal"
      aria-describedby="file-browser-description"
    >
      <Box
        width="60%"
        height="80%"
        sx={{
          position: "absolute",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          bgcolor: "background.paper",
          boxShadow: 24,
          p: 4,
          maxHeight: "90vh",
          overflowY: "auto",
          borderRadius: 3,
        }}
      >
        {FileBrowserContent}
      </Box>
    </Modal>
  ) : (
    FileBrowserContent
  );
};
