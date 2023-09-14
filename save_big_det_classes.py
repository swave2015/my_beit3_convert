# Define a function to chunk the list into multiple lines
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Read the file and extract class names
with open("bigdet_class.list", "r") as f:
    # Strip whitespace, remove number index, and replace spaces with underscores
    classes = [line.strip().split(":")[1].strip().replace(" ", "_") for line in f]

# Chunk the classes into lines with 10 (or another number) classes per line
chunked_classes = list(chunks(classes, 10))

# Format the chunked classes into the desired multiline Python tuple representation
formatted_classes = "CLASSES = (\n"
for chunk in chunked_classes:
    formatted_classes += "    " + ", ".join(f"'{cls}'" for cls in chunk) + ",\n"
formatted_classes += ")"

# Save the result to a file
with open("output_classes.py", "w") as out_file:
    out_file.write(formatted_classes)
