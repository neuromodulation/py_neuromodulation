import { useEffect, useState, useRef } from "react";
import invariant from "tiny-invariant";
import { draggable } from "@atlaskit/pragmatic-drag-and-drop/element/adapter";
import { Add, Remove } from "@mui/icons-material";

import {
  Divider,
  IconButton,
  Paper,
  List,
  ListItem,
  ListItemText,
  Stack,
  Typography,
} from "@mui/material";
import { formatKey } from "@/utils";
import { TitledBox } from "@/components";

//
const ListCard = ({ item, onButtonClick, mode }) => {
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

  let Icon = null;
  if (mode === "remove") {
    Icon = Remove;
  } else if (mode === "add") {
    Icon = Add;
  }

  return (
    <Paper elevation={1} bgcolor="background.paper" sx={{ my: 1 }} width="100%">
      <ListItem
        secondaryAction={
          <IconButton edge="end" onClick={() => onButtonClick(item)}>
            <Icon />
          </IconButton>
        }
        ref={ref}
      >
        <ListItemText primary={formatKey(item)} />
      </ListItem>
    </Paper>
  );
};

const ListContainer = ({ title, mode, items, onButtonClick }) => (
  <TitledBox title={title} sx={{ borderRadius: 3 }}>
    <List sx={{ m: 0, p: 0, width: "100%" }}>
      {items.map((item, index) => (
        <ListCard
          key={index}
          item={item}
          mode={mode}
          onButtonClick={onButtonClick}
        />
      ))}
      {items.length === 0 && (
        <ListItem>
          <ListItemText
            primary="No items available"
            sx={{ color: "text.secondary", fontStyle: "italic" }}
          />
        </ListItem>
      )}
    </List>
  </TitledBox>
);

export const OrderableLiteralListField = ({
  label,
  value = [],
  onChange,
  error,
  valid_values = [],
}) => {
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
    <Stack>
      <Typography variant="h6">{label}</Typography>

      <ListContainer
        title="Selected"
        mode="remove"
        items={selectedItems}
        onButtonClick={handleRemove}
      />
      <ListContainer
        title="Available"
        mode="add"
        items={availableItems}
        onButtonClick={handleAdd}
      />

      {error && (
        <Typography color="error" variant="caption">
          {error}
        </Typography>
      )}
    </Stack>
  );
};
