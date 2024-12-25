import re

# Define file paths
file_path = "/tmp/extract.log"  # Path to the input file
output_file = "/tmp/extract_out.log"  # Path to the output file

# Initialize variables
lines = []
current_prompt = None
previous_prompt_value = None

# Helper function to parse prompt values
def parse_prompt_value(value):
    value = value.lower().replace("k", "000")  # Convert 12K or 12k to 12000
    try:
        return int(value)
    except ValueError:
        return None

# Process the file
with open(file_path, 'r') as file:
    for line in file:
        line = line.strip()
        if line.lower().startswith("prompt ="):
            prompt_str = line.split("=")[1]
            prompt_value = parse_prompt_value(prompt_str)

            # Check if the prompt number restarts
            if previous_prompt_value is not None and prompt_value is not None and prompt_value < previous_prompt_value:
                lines.append("\n++++++++++++++++++++++++++++++++\n")
                lines.append("Prompt\t TTFT(ms)")

            previous_prompt_value = prompt_value

            # If there's a leftover prompt without TTFT, add it with N/A
            # if current_prompt:
            #     lines.append(f"{current_prompt}, TTFT = N/A")
            current_prompt = line  # Store the new prompt line
        elif line.startswith("Median TTFT (ms):"):
            if current_prompt:
                # Extract the last non-empty value from the TTFT line
                ttft_value = line.split()[-1]
                lines.append(f"{prompt_value} \t {ttft_value}")
                current_prompt = None  # Reset current prompt after pairing

# Handle any leftover prompt without a TTFT
# if current_prompt:
#     lines.append(f"{current_prompt}, TTFT = N/A")

# Write to output file
with open(output_file, 'w') as file:
    file.write("\n".join(lines) + "\n")

# Display formatted output
print("Formatted output:")
print("\n".join(lines))
