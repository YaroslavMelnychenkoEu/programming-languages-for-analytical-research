def format_code(code: str) -> str:
    keywords = ('for', 'if', 'else', 'in', ':')
    lines = code.splitlines()
    formatted_code = ""
    indent_level = 0

    for line in lines:
        stripped_line = line.strip()
        
        if stripped_line.startswith("else"):
            indent_level -= 1

        formatted_code += "    " * indent_level + stripped_line + "\n"

        if any(keyword in stripped_line for keyword in keywords) and stripped_line.endswith(":"):
            indent_level += 1

    return formatted_code

code = """
for i in range(5):
if i % 2 == 0:
print("Even")
else:
print("Odd")
"""

formatted_code = format_code(code)
print(formatted_code)