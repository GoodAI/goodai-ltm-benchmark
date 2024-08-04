import os

def read_files_in_directory(directory):
    file_contents = []

    for root, dirs, files in os.walk(directory):
        # Exclude '__pycache__' directories
        dirs[:] = [d for d in dirs if d != '__pycache__']
        
        for file in files:
            # Skip '__init__.py' files
            if file == '__init__.py':
                continue
            
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    relative_path = os.path.relpath(file_path, directory)
                    file_contents.append((relative_path, content))
                    print(f"Read file: {relative_path}")  # Debug statement
            except Exception as e:
                print(f"Failed to read file: {file_path} with error: {e}")  # Debug statement
    
    return file_contents

def format_for_llm(file_contents):
    formatted_output = ""
    
    for file_name, content in file_contents:
        formatted_output += f"### File: {file_name} ###\n"
        formatted_output += content + "\n\n"
    
    return formatted_output

def main():
    directory = './ltm'
    file_contents = read_files_in_directory(directory)
    
    if not file_contents:
        print("No file contents read.")  # Debug statement
    
    formatted_output = format_for_llm(file_contents)
    
    output_file = './ltm/txt.txt'
    with open(output_file, 'w') as f:
        f.write(formatted_output)
    
    print(f"Output written to: {output_file}")  # Debug statement

if __name__ == "__main__":
    main()
