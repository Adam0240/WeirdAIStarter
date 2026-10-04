"""
Instruction fine-tuning helpers for Weird AI.
"""

import torch


def extract_response(generated_text, prompt_text):
    """
    Remove the prompt from generated text and return only the response.
    """

    # Remove prompt_text from the beginning of generated_text.
    response = generated_text.removeprefix(prompt_text)

    # Strip extra whitespace.
    return response.strip()

def save_instruction_model(model, path):
    """
    Save instruction fine-tuned model weights.
    """

    # Use torch.save with model.state_dict().
    torch.save(model.state_dict(), path)

    raise NotImplementedError("Implement save_instruction_model.")


def load_instruction_model(model, path, device):
    """
    Load instruction fine-tuned model weights.
    """

    # Use torch.load and model.load_state_dict.
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)

