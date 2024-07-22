#!/bin/bash

# Loop through all .jsx files
for file in *.jsx; do
    # Get the component name (filename without extension)
    component_name="${file%.jsx}"
    
    # Create a new directory for the component
    mkdir -p "$component_name"
    
    # Move the .jsx file into the new directory
    git mv "$file" "$component_name/"
    
    # Create a new .module.css file
    touch "$component_name/$component_name.module.css"
    
    # Create an index.js file
    echo "export { $component_name } from './$component_name';" > "$component_name/index.js"
        
    echo "Processed $component_name"
done

echo "All components have been processed."