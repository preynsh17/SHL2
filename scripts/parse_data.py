import json
import regex as re
import logging

# [cite_start]Mapping from the PDF [cite: 59-67]
TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations"
}

# --- Define Regex Patterns to find our data ---

# Pattern 1: Find the "Description" block
# Looks for "Description" followed by the text until it hits "Job levels" or "Languages"
desc_pattern = re.compile(
    r'Description\s+(.*?)\s+(Job levels|Languages|Assessment length)',
    re.DOTALL | re.IGNORECASE
)

# Pattern 2: Find the "Duration"
# Looks for "Approximate Completion Time in minutes = " followed by numbers
duration_pattern = re.compile(
    r'Approximate Completion Time in minutes\s*=\s*(\d+)',
    re.IGNORECASE
)

# Pattern 3: Find the "Test Type" letters
# Looks for "Test Type: " followed by space-separated letters (e.g., "C P A B")
test_type_pattern = re.compile(
    r'Test Type:\s+([A-Z\s]+?)\s+(Remote Testing|Downloads|Accelerate)',
    re.DOTALL | re.IGNORECASE
)

# Pattern 4: Find "Remote Testing"
# Just checks if the string "Remote Testing:" exists
remote_pattern = re.compile(r'Remote Testing:', re.IGNORECASE)


def parse_raw_assessment(line_data):
    """
    Takes one line (as a dict) from the .jsonl file
    and extracts the clean data from its 'raw_text' field.
    """
    
    # Get the fields we already have
    name = line_data.get('name')
    url = line_data.get('url')
    raw_text = line_data.get('raw_text', '')

    if not raw_text:
        return None

    # --- Initialize clean data fields ---
    clean_description = ""
    clean_duration = None
    clean_test_types = []
    clean_remote = "No" # Default to "No" [cite: 120, 123]
    
    # --- Run Regex ---
    
    # Find Description
    desc_match = desc_pattern.search(raw_text)
    if desc_match:
        clean_description = desc_match.group(1).strip().replace('\n', ' ')
    
    # Find Duration
    duration_match = duration_pattern.search(raw_text)
    if duration_match:
        try:
            clean_duration = int(duration_match.group(1))
        except ValueError:
            clean_duration = None # Set to None if conversion fails
            
    # Find Remote Support
    if remote_pattern.search(raw_text):
        clean_remote = "Yes"
        
    # Find Test Types
    test_type_match = test_type_pattern.search(raw_text)
    if test_type_match:
        letters_str = test_type_match.group(1)
        letters = letters_str.split()
        for letter in letters:
            if letter in TEST_TYPE_MAP:
                clean_test_types.append(TEST_TYPE_MAP[letter])

    # --- Fallback Logic ---
    # If regex fails, try to use the (less accurate) top-level fields
    
    # If we couldn't find ANY test types in raw_text, use the top-level one
    if not clean_test_types and line_data.get('test_type'):
        clean_test_types.append(line_data['test_type'])
        
    # If description is still empty, use a placeholder
    if not clean_description:
        clean_description = f"Assessment for {name}."

    # Build the final clean dictionary
    # [cite_start]This structure matches the API requirements [cite: 126, 127]
    clean_data = {
        "name": name,
        "url": url,
        "description": clean_description,
        "test_type": clean_test_types,
        "duration": clean_duration,
        "adaptive_support": "No", # Not in raw text, so we default to "No" [cite: 120]
        "remote_support": clean_remote
    }
    
    return clean_data


def main():
    """
    Main function to read 'assessments.jsonl',
    parse each line, and write to 'assessments_clean.json'.
    """
    input_file = 'assessments.jsonl'
    output_file = 'assessments_clean.json'
    
    clean_assessments = []
    
    print(f"Starting to parse {input_file}...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            for i, line in enumerate(f_in):
                try:
                    line_data = json.loads(line)
                    clean_data = parse_raw_assessment(line_data)
                    
                    # Only add if parsing was successful and it's a valid assessment
                    if clean_data and clean_data.get('duration') is not None:
                        clean_assessments.append(clean_data)
                    else:
                        logging.warning(f"Skipping line {i+1} (Name: {line_data.get('name')}): Could not parse duration or other critical fields.")
                        
                except json.JSONDecodeError:
                    logging.error(f"Error decoding JSON on line {i+1}. Skipping.")
                except Exception as e:
                    logging.error(f"Error processing line {i+1} (Name: {line_data.get('name')}): {e}")

        print(f"Successfully parsed {len(clean_assessments)} assessments.")

        # Write the clean data to our new database file
        with open(output_file, 'w', encoding='utf-8') as f_out:
            json.dump(clean_assessments, f_out, indent=2)
            
        print(f"Successfully created clean database at {output_file}")

    except FileNotFoundError:
        print(f"ERROR: Input file '{input_file}' not found.")
        print("Please make sure 'assessments.jsonl' is in the same folder.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()