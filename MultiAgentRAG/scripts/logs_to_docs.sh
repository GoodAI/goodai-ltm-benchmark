#!/bin/bash

# Define the logs directory
LOGS_DIR="./logs"
OUTPUT_DIR="./text_docs"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Iterate over all log files in the logs directory
for log_file in "$LOGS_DIR"/*.log; 
do
  # Get the base name of the log file (without the directory path)
  base_name=$(basename "$log_file")
  
  # Remove the .log extension and append .txt
  output_file="${base_name%.log}.txt"
  
  # Copy the log file to the output directory with the new extension
  cp "$log_file" "$OUTPUT_DIR/$output_file"
  
  echo "Converted $log_file to $OUTPUT_DIR/$output_file"
done

echo "All log files have been converted to text documents."
