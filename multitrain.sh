#!/bin/bash

# Check if the input file is provided
if [ -z "$1" ]; then
  echo "Usage: $0 input_file"
  exit 1
fi

INPUT_FILE="$1"

# Check if the file exists
if [ ! -f "$INPUT_FILE" ]; then
  echo "Error: File '$INPUT_FILE' not found."
  exit 1
fi

# Read the file line by line
while IFS= read -r line || [ -n "$line" ]; do
  # Skip empty lines or lines starting with '#'
  if [[ -z "$line" || "$line" =~ ^# ]]; then
    continue
  fi

  # Check if the line matches the pattern: script_name n
  if [[ "$line" =~ ^([^\ ]+)[[:space:]]+([0-9]+)$ ]]; then
    SCRIPT_NAME="${BASH_REMATCH[1]}"
    NUM_LINES="${BASH_REMATCH[2]}"

    echo "Processing script: $SCRIPT_NAME with $NUM_LINES set(s) of flags"

    # Initialize an array to hold flags
    FLAGS_ARRAY=()

    # Read the next NUM_LINES lines as flags
    for (( i=0; i<NUM_LINES; i++ )); do
      if ! IFS= read -r flag_line || [ -z "$flag_line" ]; then
        echo "Error: Expected $NUM_LINES lines of flags for $SCRIPT_NAME, but encountered end of file."
        exit 1
      fi

      FLAGS_ARRAY+=("$flag_line")
    done

    # Run the script NUM_LINES times with the corresponding flags
    for flag_line in "${FLAGS_ARRAY[@]}"; do
      echo "Running $SCRIPT_NAME with flags: $flag_line"
      sh $SCRIPT_NAME $flag_line
    done

  else
    echo "Error: Expected script name and number of lines, but got: $line"
    exit 1
  fi

done < "$INPUT_FILE"
