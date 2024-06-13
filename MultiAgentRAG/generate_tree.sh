#!/bin/bash

# Generate tree structure using find and sed, excluding specified files and directories
find . \( -name '__pycache__' -o -name '.git' \) -prune -o -type f ! \( -name 'tree.txt' -o -name 'generate_tree.sh' -o -name 'README' -o -name '*.ipynb' \) -print | sed -e 's|[^/]*/| |g' -e 's|^|   |' > tree.txt

# Append file contents, excluding specified files
echo -e "\nFile Contents:\n" >> tree.txt
find . \( -name 'tree.txt' -o -name 'generate_tree.sh' -o -name 'README' -o -name '*.ipynb' -o -path '*/__pycache__/*' -o -path '*/.git/*' \) -prune -o -type f -print0 | xargs -0 -I {} sh -c 'echo -e "\nFile: {}\n"; cat "{}"' >> tree.txt
