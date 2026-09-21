"""
Training utilities for Weird AI.

This module contains the training loop used to pretrain the Weird AI model
on unlabeled lyric data.
"""

import torch

from weird_ai.losses import calc_loss_batch, calc_loss_loader
from weird_ai.generation import generate_and_print_sample


def evaluate_model(
    model,
    train_loader,
    val_loader,
    device,
    eval_iter
):
    """
    Evaluate the model on a limited number of training and validation batches.

    Args:
        model: The Weird AI model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        device: CPU or CUDA device.
        eval_iter: Number of batches to use for evaluation.

    Returns:
        A tuple containing:
            train_loss: Average training loss.
            val_loss: Average validation loss.
    """

    model.eval()

    # TODO:
    # 1. Disable gradient tracking with torch.no_grad().
    with torch.no_grad():

        # 2. Calculate training loss using calc_loss_loader.
        train_loss = calc_loss_loader(
            train_loader,
            model,
            device,
            num_batches=eval_iter
        )

        # 3. Calculate validation loss using calc_loss_loader.
        val_loss = calc_loss_loader(
            val_loader,
            model,
            device,
            num_batches=eval_iter
        )

    # 4. Return both losses.
    return train_loss, val_loss


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
    context_size
):
    """
    Train the Weird AI model using a basic PyTorch training loop.

    Args:
        model: The Weird AI model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: PyTorch optimizer.
        device: CPU or CUDA device.
        num_epochs: Number of epochs to train.
        eval_freq: How often to evaluate, in training steps.
        eval_iter: Number of batches to use during evaluation.
        start_context: Prompt used to generate sample text.
        tokenizer: Tokenizer used for text generation.
        context_size: Maximum context size for generation.

    Returns:
        A tuple containing:
            train_losses: List of recorded training losses.
            val_losses: List of recorded validation losses.
            track_tokens_seen: List of token counts seen at each evaluation point.
    """

    train_losses = []
    val_losses = []
    track_tokens_seen = []

    tokens_seen = 0
    global_step = -1

    # TODO:
    # Move model to the selected device.
    model = model.to(device)

    for epoch in range(num_epochs):

        # TODO:
        # Put model in training mode.
        model.train()

        for input_batch, target_batch in train_loader:

            # TODO:
            # 1. Reset gradients with optimizer.zero_grad().
            optimizer.zero_grad()

            # 2. Calculate loss for this batch.
            loss = calc_loss_batch(
                input_batch,
                target_batch,
                model,
                device
            )

            # 3. Run backpropagation with loss.backward().
            loss.backward()

            # 4. Update model weights with optimizer.step().
            optimizer.step()

            # 5. Update tokens_seen.
            tokens_seen += input_batch.numel()

            # 6. Update global_step.
            global_step += 1

            # TODO:
            # If global_step is divisible by eval_freq:
            #   1. Evaluate the model.
            #   2. Store train loss, validation loss, and tokens seen.
            #   3. Print progress.

            if global_step % eval_freq == 0:

                train_loss, val_loss = evaluate_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    eval_iter
                )

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(
                    f"Epoch {epoch + 1}, "
                    f"Step {global_step}: "
                    f"Train loss {train_loss:.3f}, "
                    f"Validation loss {val_loss:.3f}"
                )

                # Return to training mode after evaluation
                model.train()

        # TODO:
        # At the end of each epoch, generate and print a sample.
        # This helps visually inspect whether the model is improving.

        generate_and_print_sample(
            model=model,
            tokenizer=tokenizer,
            device=device,
            start_context=start_context,
            context_size=context_size
        )

    return train_losses, val_losses, track_tokens_seen


def save_checkpoint(
    model,
    optimizer,
    epoch,
    train_losses,
    val_losses,
    track_tokens_seen,
    checkpoint_path
):
    """
    Save model and optimizer state so training can continue later.

    Args:
        model: The Weird AI model.
        optimizer: PyTorch optimizer.
        epoch: Current epoch.
        train_losses: Recorded training losses.
        val_losses: Recorded validation losses.
        track_tokens_seen: Recorded token counts.
        checkpoint_path: Path where checkpoint should be saved.
    """

    # TODO:
    # Use torch.save to save a dictionary containing:
    #   model_state_dict
    #   optimizer_state_dict
    #   epoch
    #   train_losses
    #   val_losses
    #   track_tokens_seen

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "track_tokens_seen": track_tokens_seen
        },
        checkpoint_path
    )


def load_checkpoint(
    model,
    optimizer,
    checkpoint_path,
    device
):
    """
    Load model and optimizer state from a checkpoint.

    Args:
        model: The Weird AI model.
        optimizer: PyTorch optimizer.
        checkpoint_path: Path to checkpoint file.
        device: CPU or CUDA device.

    Returns:
        A dictionary containing checkpoint metadata.
    """

    # TODO:
    # 1. Load the checkpoint with torch.load.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    # 2. Restore the model state.
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # 3. Restore the optimizer state.
    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    # 4. Return metadata such as epoch and loss history.
    metadata = {
        "epoch": checkpoint["epoch"],
        "train_losses": checkpoint["train_losses"],
        "val_losses": checkpoint["val_losses"],
        "track_tokens_seen": checkpoint["track_tokens_seen"]
    }

    return metadata