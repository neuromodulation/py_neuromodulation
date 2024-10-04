import React, { useState, useEffect } from "react";
import { getBackendURL } from "@/utils/getBackendURL";
import {
  Paper,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from "@mui/material";
import {
  Folder,
  Star,
  Computer,
  Download,
  InsertDriveFile,
} from "@mui/icons-material";

export const QuickAccessSidebar = ({ onItemClick }) => {
  const [quickAccessItems, setQuickAccessItems] = useState([]);

  useEffect(() => {
    fetchQuickAccessItems();
  }, []);

  const fetchQuickAccessItems = async () => {
    try {
      const response = await fetch(getBackendURL("/api/quick-access"));
      if (!response.ok) throw new Error("Failed to fetch quick access items");
      const data = await response.json();
      setQuickAccessItems(data.items);
    } catch (error) {
      console.error("Error fetching quick access items:", error);
    }
  };

  const getIconForItem = (item) => {
    switch (item.type) {
      case "folder":
        return <Folder fontSize="small" />;
      case "drive":
        return <Computer fontSize="small" />;
      case "download":
        return <Download fontSize="small" />;
      case "file":
        return <InsertDriveFile fontSize="small" />;
      default:
        return <Star fontSize="small" />;
    }
  };

  return (
    <TableContainer
      component={Paper}
      sx={{
        flexGrow: 1,
        display: "flex",
        flexDirection: "column",
        maxHeight: 400,
        overflowX: "hidden",
        overflowY: "auto",
        scrollbarWidth: "thin",
        scrollbarColor: "rgba(0,0,0,.1) transparent",
      }}
    >
      <Table size="small" stickyHeader>
        <TableHead>
          <TableRow sx={{ height: "2.5rem" }}>
            <TableCell
              sx={{
                borderBottom: "1px solid",
                borderColor: "divider",
              }}
            >
              <Typography variant="body" sx={{ fontWeight: "bold" }}>
                Quick Access
              </Typography>
            </TableCell>
          </TableRow>
        </TableHead>
        <TableBody sx={{ overflow: "auto" }}>
          {quickAccessItems.map((item, index) => (
            <TableRow
              key={index}
              hover
              onClick={() => onItemClick(item.path)}
              sx={{ cursor: "pointer" }}
            >
              <TableCell
                sx={{ py: 0.5, display: "flex", alignItems: "center", gap: 1 }}
              >
                {getIconForItem(item)}
                <Typography variant="body2" noWrap>
                  {item.name}
                </Typography>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};
