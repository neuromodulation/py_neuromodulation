import { useEffect, useState, useRef } from "react";
import invariant from "tiny-invariant";
import { draggable } from "@atlaskit/pragmatic-drag-and-drop/element/adapter";
import { Add, Remove } from "@mui/icons-material";

import {
  Divider,
  IconButton,
  List,
  ListItem,
  ListItemText,
  Stack,
  Typography,
} from "@mui/material";

export const OrderableLiteralListField = ({
  label,
  value = [],
  onChange,
  error,
  valid_values = [],
}) => {
  const ListCard = ({ key, item }) => {
    const ref = useRef(null);
    const [dragging, setDragging] = useState(false);

    useEffect(() => {
      const el = ref.current;
      invariant(el);

      return draggable({
        element: el,
        onDragStart: () => setDragging(true),
        onDrop: () => setDragging(false),
      });
    }, []);

    return (
      <ListItem
        key={key}
        secondaryAction={
          <IconButton edge="end" onClick={() => handleRemove(item)}>
            <Remove />
          </IconButton>
        }
        ref={ref}
      >
        <ListItemText primary={item} />
      </ListItem>
    );
  };

  // Create sets for faster lookup
  const selectedSet = new Set(value);

  // Filter valid_values into selected and available arrays
  const selectedItems = valid_values.filter((item) => selectedSet.has(item));
  const availableItems = valid_values.filter((item) => !selectedSet.has(item));

  const handleAdd = (item) => {
    const newValue = [...value, item];
    onChange(newValue);
  };

  const handleRemove = (item) => {
    const newValue = value.filter((val) => val !== item);
    onChange(newValue);
  };

  return (
    <Stack spacing={2}>
      <Typography variant="h6">{label}</Typography>

      <div>
        <Typography variant="subtitle1" color="primary" sx={{ mb: 1 }}>
          Selected Items
        </Typography>
        <List>
          {selectedItems.map((item, index) => (
            <ListCard key={index} item={item} />
          ))}
          {selectedItems.length === 0 && (
            <ListItem>
              <ListItemText
                primary="No items selected"
                sx={{ color: "text.secondary", fontStyle: "italic" }}
              />
            </ListItem>
          )}
        </List>
      </div>

      <Divider />

      <div>
        <Typography variant="subtitle1" color="primary" sx={{ mb: 1 }}>
          Available Items
        </Typography>
        <List>
          {availableItems.map((item) => (
            <ListItem
              key={item}
              secondaryAction={
                <IconButton edge="end" onClick={() => handleAdd(item)}>
                  <Add />
                </IconButton>
              }
            >
              <ListItemText primary={item} />
            </ListItem>
          ))}
          {availableItems.length === 0 && (
            <ListItem>
              <ListItemText
                primary="No items available"
                sx={{ color: "text.secondary", fontStyle: "italic" }}
              />
            </ListItem>
          )}
        </List>
      </div>

      {error && (
        <Typography color="error" variant="caption">
          {error}
        </Typography>
      )}
    </Stack>
  );
};
