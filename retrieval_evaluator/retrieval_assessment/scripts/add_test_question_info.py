import json

# Hardcoded file paths
input_file_path = r'C:\Users\fkgde\Desktop\GoodAI\__FULLTIME\LTM-Benchmark\goodai-ltm-benchmark\MLP_LTM\retrieval_assessment\reference_data\comparison_data_reference_2-2.json'
output_file_path = r'C:\Users\fkgde\Desktop\GoodAI\__FULLTIME\LTM-Benchmark\goodai-ltm-benchmark\MLP_LTM\retrieval_assessment\reference_data\comparison_data_reference_enhanced_2-2.json'

# Function to add "test" and "is_scored_question" fields
def add_fields(data):
    for entry in data:
        entry["test"] = ""
        entry["is_scored_question"] = "no"
    return data

# Read data from input file
with open(input_file_path, 'r') as input_file:
    data = json.load(input_file)

# Modify data
modified_data = add_fields(data)

# Write modified data to output file
with open(output_file_path, 'w') as output_file:
    json.dump(modified_data, output_file, indent=2)

print("Modification complete. Check the output file.")
