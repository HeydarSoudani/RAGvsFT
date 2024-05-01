import re

def extract_number(text):
    # Regex to find the first occurrence of one or more digits
    match = re.search(r'\d+', text)
    if match:
        return int(match.group(0))
    else:
        return None


text = """
The correct answer is Answer 2.
"""


correct_answer_number = extract_number(text)
if correct_answer_number is not None:
    print("The correct answer is:", correct_answer_number)
else:
    print("No correct answer number found in the text.")
