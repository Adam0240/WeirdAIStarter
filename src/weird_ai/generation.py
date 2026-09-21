"""
Text generation utilities for Weird AI.

This module contains helper functions that convert between text and token IDs
and generate new text from a trained or partially trained model.
"""

import torch


def text_to_token_ids(text, tokenizer):
    """
    Convert text into a tensor of token IDs.

    Args:
        text: The input prompt as a string.
        tokenizer: The tokenizer object.

    Returns:
        A tensor of shape (1, num_tokens).
    """

    encoded = tokenizer.encode(text)
    token_ids = torch.tensor(
        encoded,
        dtype=torch.long
    )

    token_ids = token_ids.unsqueeze(0)

    return token_ids


def token_ids_to_text(token_ids, tokenizer):
    """
    Convert token IDs back into text.

    Args:
        token_ids: Tensor of token IDs.
        tokenizer: The tokenizer object.

    Returns:
        The decoded text as a string.
    """

    # Remove batch dimension if present
    if token_ids.dim() == 2:
        token_ids = token_ids.squeeze(0)

    # Move to CPU and convert to Python list
    token_ids = token_ids.cpu().tolist()

    # Decode tokens
    return tokenizer.decode(token_ids)

def generate_text_simple(model, input_ids, max_new_tokens, context_size):
    """
    Generate text one token at a time using greedy decoding.

    Args:
        model: The Weird AI model.
        input_ids: Tensor of shape (batch_size, num_tokens).
        max_new_tokens: Number of new tokens to generate.
        context_size: Maximum number of tokens the model can consider.

    Returns:
        Tensor containing the original input IDs plus generated token IDs.
    """

    for _ in range(max_new_tokens):

        # Keep only tokens that fit inside the context window
        input_crop = input_ids[:, -context_size:]

        # Get model predictions
        with torch.no_grad():
            logits = model(input_crop)

        # Get logits for the final token position
        logits = logits[:, -1, :]

        # Select token with highest score
        next_token = torch.argmax(
            logits,
            dim=-1,
            keepdim=True
        )

        # Add predicted token to sequence
        input_ids = torch.cat(
            (input_ids, next_token),
            dim=1
        )

    return input_ids


def generate_and_print_sample(model, tokenizer, device, start_context, context_size, max_new_tokens=50):
    """
    Generate and print a sample text output.

    This is useful during training so students can visually inspect whether
    the model is improving.

    Args:
        model: The Weird AI model.
        tokenizer: The tokenizer object.
        device: CPU or CUDA device.
        start_context: Prompt text.
        context_size: Maximum number of tokens the model can consider.
        max_new_tokens: Number of tokens to generate.
    """

    model.eval()

    # Convert prompt into token IDs
    token_ids = text_to_token_ids(
        start_context,
        tokenizer
    )

    # Move token IDs to device
    token_ids = token_ids.to(device)

    # Generate new tokens
    generated_ids = generate_text_simple(
        model=model,
        input_ids=token_ids,
        max_new_tokens=max_new_tokens,
        context_size=context_size
    )

    # Convert tokens back into text
    generated_text = token_ids_to_text(
        generated_ids,
        tokenizer
    )

    print(generated_text)
