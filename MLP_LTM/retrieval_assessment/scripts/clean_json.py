import json

def clean_json(data):
    if isinstance(data, dict):
        return {k: clean_json(v) for k, v in data.items() if k not in ["id", "timestamp"]}
    elif isinstance(data, list):
        return [clean_json(item) for item in data]
    else:
        return data

# Read the input JSON file
with open('comparison_data_reference.json', 'r') as file:
    json_data = json.load(file)

# Clean the JSON data
cleaned_data = clean_json(json_data)

# Write the cleaned data to a new JSON file
with open('sanitized_comparison_data_reference.json', 'w') as file:
    json.dump(cleaned_data, file, indent=2)

print("JSON cleaning complete. Output saved to 'cleaned_output.json'.")