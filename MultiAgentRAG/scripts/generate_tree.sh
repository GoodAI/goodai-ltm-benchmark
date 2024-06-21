#!/bin/bash

# File where the tree structure will be stored
OUTPUT_FILE="tree.txt"

# Clear the previous contents of the output file
> "$OUTPUT_FILE"

# Generate tree structure identical to the tree command output
tree >> "$OUTPUT_FILE"

# Add a note about the exclusions for appended file contents
echo -e "\nNote: The following list of files and directories are excluded only from the appended file contents section:\n" >> "$OUTPUT_FILE"
echo -e "__pycache__, .git, .env, text_docs, scripts, logs, tree.txt, generate_tree.sh, README, *.ipynb, *.pdf, *.db\n" >> "$OUTPUT_FILE"

# Append file contents, excluding specified files and directories
echo -e "File Contents:\n" >> "$OUTPUT_FILE"
find . \( -name 'tree.txt' -o -name 'generate_tree.sh' -o -name 'README' -o -name '*.ipynb' -o -name '*.pdf' -o -name '*.db' -o -name '.env' -o -path '*/__pycache__/*' -o -path '*/.git/*' -o -path '*/text_docs/*' -o -path '*/scripts/*' -o -path '*/logs/*' \) -prune -o -type f -print0 | xargs -0 -I {} sh -c 'echo -e "\nFile: {}\n"; cat "{}"' >> "$OUTPUT_FILE"

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
