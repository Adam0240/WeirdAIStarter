"""
Reasoning prompt helpers for Weird AI.

This module introduces the first lightweight reasoning-related utilities for
Weird AI. These functions do not train a reasoning model yet. Instead, they
help students explore how prompt structure can encourage a model to produce
intermediate reasoning steps before generating a final answer.

Lesson 8 connects the Weird AI parody assistant to the reasoning model portion
of the course by focusing on inference-time reasoning prompts.
"""


def build_reasoning_prompt(user_prompt):
    """
    Convert a normal user prompt into a general reasoning prompt.

    The returned prompt should encourage the model to think through the task
    before giving its final answer.

    Args:
        user_prompt: The original user request as a string.

    Returns:
        A formatted prompt string that includes the original task and asks for
        step-by-step reasoning.
    """
    return (
    # 1. Include a short role statement such as "You are Weird AI."
    "You are a Weird AI.\n"
    # 2. Include the phrase "Think step-by-step."
    "Think step-by-step.\n"
    # 3. Include the original user_prompt.
    f"Question: {user_prompt}\n"
    # 4. Include a clear "Final Answer:" section.
    "Final Answer:\n"
    )
    

def build_parody_reasoning_prompt(topic):
    """
    Build a reasoning prompt specifically for parody lyric generation.

    This prompt should guide Weird AI through a simple creative planning process
    before asking it to write the final parody lyrics.

    Args:
        topic: The parody topic or user request as a string.

    Returns:
        A formatted prompt string with step-by-step parody planning sections.
    """
    return(
    # 1. The phrase "Think step-by-step."
    "Think step-by-step.\n"
    # 2. The topic provided by the user.
    f"Topic: {topic}.\n"
    # 3. At least these planning steps:
    #    - Identify the topic.
    "Identify the topic.\n"
    #    - Identify funny ideas or exaggerations.
    "Identify funny ideas or exaggerations.\n"
    #    - Identify possible rhymes.
    "Identify possible rhymes.\n"
    #    - Write the final parody lyrics.
    "Write final parody lyrics\n"
    # 4. A clear "Final Parody:" section.
    "Final Parody:\n"
    )


def build_closed_world_prompt(premises, question):
    """
    Build a prompt for closed-world reasoning.

    Closed-world reasoning means the model should answer using only the facts
    explicitly provided in the prompt. It should not use outside knowledge.

    Args:
        premises: A list of premise strings.
        question: The question to answer from those premises.

    Returns:
        A formatted prompt string for closed-world reasoning.
    """
    return (
        # 1. Tell the model to use only the provided premises.
        "Use only the provided premises.\n"
        # 2. Include each premise in a readable list.
        "Premises:\n"
        + "\n".join(f"- {premise}" for premise in premises)
        + "\n"
        # 3. Include the question.
        + f"Question: {question}\n"
        # 4. Include reasoning and answer sections.
        + "Reasoning:\n"
        + "Answer:\n"

    )

def build_open_world_prompt(premises, question):
    """
    Build a prompt for open-world reasoning.

    Open-world reasoning allows the model to consider both the provided premises
    and relevant background knowledge. This can reveal contradictions between
    the prompt and the model's background knowledge.

    Args:
        premises: A list of premise strings.
        question: The question to answer.

    Returns:
        A formatted prompt string for open-world reasoning.
    """
    return (
        # 1. Tell the model it may use relevant background knowledge.
        "Use the relevant background knowledge.\n"
        # 2. Ask it to check whether the premises conflict with known facts.
        "Check if the premises conflict with known facts.\n"
        # 3. Include each premise in a readable list.
        "Premises:\n"
        + "\n".join(f"- {premise}" for premise in premises)
        + "\n"
        # 4. Include the question.
        + f"Question: {question}\n"
        # 5. Include reasoning and answer sections.
        + "Reasoning:\n"
        + "Answer:\n"
    )



