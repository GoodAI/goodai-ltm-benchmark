#!/bin/bash
# reset_logs.sh
# This script resets all log files in the specified directory.

# Directory containing the log files
LOG_DIR="./logs"

# Check if the directory exists
if [ ! -d "$LOG_DIR" ]; then
    echo "Directory $LOG_DIR does not exist."
    exit 1
fi

# Iterate over all log files in the directory
for log_file in "$LOG_DIR"/*.log; 
do
    # Check if the file exists
    if [ -f "$log_file" ]; then
        # Truncate the file to zero length
        > "$log_file"
        echo "Reset $log_file"
    else
        echo "No log files found in $LOG_DIR"
    fi
done

echo "All log files have been reset."
