"""
Instruction data formatting utilities for Weird AI.
"""


def format_input(entry):
    """
    Format one instruction example using an Alpaca-style prompt.
    This function does NOT include the response.
    """

    # 1. Build the standard instruction text.
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    )

    # 2. Add the ### Instruction section.
    instruction_text += f"\n\n### Instruction:\n{entry['instruction']}"

    # 3. Add the ### Input section only when entry["input"] is not empty.
    if entry["input"]:
        instruction_text += f"\n\n### Input:\n{entry['input']}"

    # 4. Return the complete prompt.
    return instruction_text


def format_response(entry):
    """
    Format the expected response section.
    """

    return f"\n\n### Response:\n{entry['output']}"


def format_full_example(entry):
    """
    Format an entire instruction-response example.
    """

    # Combine format_input(entry) and format_response(entry).
    return format_input(entry) + format_response(entry)

def validate_instruction_entry(entry):
    """
    Validate that an instruction dataset entry has instruction, input, and output fields.
    """

    # Check for instruction, input, and output keys.
    if not isinstance(entry, dict):
        return False

    for key in ("instruction", "input", "output"):
        if key not in entry or not isinstance(entry[key], str):
            return False
        
    # Verify that instruction and output are not empty.
    if not entry["instruction"].strip() or not entry["output"].strip():
        return False

    return True 
