#!/bin/bash

# File where the tree structure will be stored
OUTPUT_FILE="tree.txt"

# Clear the previous contents of the output file
> "$OUTPUT_FILE"

# Generate tree structure using find and sed, excluding specified files and directories
find . \( -name '__pycache__' -o -name '.git' -o -name '.env' \) -prune -o -type f ! \( -name 'tree.txt' -o -name 'generate_tree.sh' -o -name 'README' -o -name '*.ipynb' \) -print | sed -e 's|[^/]*/| |g' -e 's|^|   |' >> "$OUTPUT_FILE"

# Append file contents, excluding specified files
echo -e "\nFile Contents:\n" >> "$OUTPUT_FILE"
find . \( -name 'tree.txt' -o -name 'generate_tree.sh' -o -name 'README' -o -name '*.ipynb' -o -name '.env' -o -path '*/__pycache__/*' -o -path '*/.git/*' \) -prune -o -type f -print0 | xargs -0 -I {} sh -c 'echo -e "\nFile: {}\n"; cat "{}"' >> "$OUTPUT_FILE"

# Copy the output file contents to the clipboard
if command -v xclip &> /dev/null; then
    # Use xclip for Linux
    xclip -selection clipboard < "$OUTPUT_FILE"
elif command -v pbcopy &> /dev/null; then
    # Use pbcopy for macOS
    pbcopy < "$OUTPUT_FILE"
else
    echo "Clipboard copy command not found. Please install xclip (Linux) or pbcopy (macOS) to enable this feature."
fi

echo "Tree structure and file contents have been written to $OUTPUT_FILE and copied to the clipboard."
