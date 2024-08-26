import json
import time
from datetime import datetime

def parse_timestamp(timestamp):
    if isinstance(timestamp, int):
        return timestamp
    elif isinstance(timestamp, str):
        try:
            # Parse string timestamp to epoch time
            dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            return int(time.mktime(dt.timetuple()))
        except ValueError as e:
            print(f"Error parsing timestamp: {timestamp}, {e}")
            return None
    return None

def clean_json(data, threshold_timestamp):
    parsed_threshold = parse_timestamp(threshold_timestamp)

    for entry in data:
        entry_timestamp = parse_timestamp(entry['timestamp'])
        if entry_timestamp and entry_timestamp > parsed_threshold:
            entry['memories'] = [
                memory for memory in entry['memories']
                if parse_timestamp(memory['timestamp']) and parse_timestamp(memory['timestamp']) >= parsed_threshold
            ]
    return data

def main():
    input_file = r'C:\Users\fkgde\Desktop\GoodAI\__FULLTIME\LTM-Benchmark\goodai-ltm-benchmark\MLP_LTM\retrieval_assessment\comparison_data_reference_2-2.json'  # Hardcoded input file path
    output_file = r'C:\Users\fkgde\Desktop\GoodAI\__FULLTIME\LTM-Benchmark\goodai-ltm-benchmark\MLP_LTM\retrieval_assessment\comparison_data_reference_2-2_time_trimmed.json'  # Hardcoded output file path

    # Read the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Specify the threshold timestamp
    threshold_timestamp = 1723031934

    # Clean the JSON data
    cleaned_data = clean_json(data, threshold_timestamp)

    # Write the cleaned JSON data to an output file
    with open(output_file, 'w') as f:
        json.dump(cleaned_data, f, indent=2)

    print(f"Cleaned data has been written to {output_file}")

if __name__ == "__main__":
    main()
