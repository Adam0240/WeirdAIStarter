"""
Self-refinement utilities for Weird AI.

This module connects Chapter 5 of Build a Reasoning Model from Scratch to the
Weird AI project.

Chapter 4 added best-of-N generation:

    generate several candidates -> evaluate each -> select the best

Chapter 5 adds self-refinement:

    generate one candidate -> evaluate -> ask for a revision -> evaluate again
    -> keep the better version -> repeat

The goal is to improve output quality during inference without retraining the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


GenerationFunction = Callable[..., str]
EvaluationFunction = Callable[[str], dict]


@dataclass
class RefinementStep:
    """Store information about one refinement attempt."""

    iteration: int
    prompt: str
    previous_text: str
    revised_text: str
    previous_score: float
    revised_score: float
    accepted: bool
    evaluation: dict = field(default_factory=dict)


def get_score(evaluation: dict, score_key: str = "overall_score") -> float:
    """Extract a numeric score from an evaluation dictionary."""

    # 1. Read score_key from the evaluation dictionary.
    score = evaluation.get(score_key)

    # 2. Convert it to a float.
    try:
        return float(score)

    # 3. If the key is missing or cannot be converted, return 0.0.
    except (TypeError, ValueError):
        return 0.0

def build_refinement_prompt(
    original_prompt: str,
    current_text: str,
    evaluation: dict,
    *,
    focus: Optional[list[str]] = None,
    preserve: Optional[list[str]] = None,
    score_key: str = "overall_score",
) -> str:
    """Build a prompt asking the model to improve a generated parody."""

    # 1. Include the original prompt or topic.
    prompt = f"Original request:\n{original_prompt}\n\n"

    # 2. Include the current lyrics.
    prompt += f"Current lyrics:\n{current_text}\n\n"

    # 3. Include the current overall score.
    score = get_score(evaluation, score_key)
    prompt += f"Current score: {score}\n\n"

    # 4. Include any available evaluation details, such as rhyme, syllables, and structure.
    prompt += "Evaluation details:\n"
    for key, value in evaluation.items():
        if key != score_key:
            prompt += f"- {key}: {value}\n"

    # 5. Include focus items, such as stronger rhymes or more consistent syllables.
    if focus:
        prompt += "\nFocus on improving:\n"
        for item in focus:
            prompt += f"- {item}\n"

    # 6. Include preserve items, such as topic, tone, and approximate length.
    if preserve:
        prompt += "\nPreserve the following:\n"
        for item in preserve:
            prompt += f"- {item}\n"

    # 7. Ask the model to return only the revised lyrics.
    #
    # Keep the prompt specific. "Make it better" is too vague.
    prompt += (
        "\nRevise the parody lyrics using the evaluation feedback. "
        "Improve the rhymes, syllable consistency, and structure "
        "while maintaining the original topic. "
        "Return only the revised lyrics without any explanations."
    )

    return prompt

def refine_once(
    original_prompt: str,
    current_text: str,
    generation_func: GenerationFunction,
    evaluation_func: EvaluationFunction,
    *,
    temperature: float = 0.7,
    top_p: Optional[float] = 0.9,
    max_new_tokens: int = 160,
    score_key: str = "overall_score",
    focus: Optional[list[str]] = None,
    preserve: Optional[list[str]] = None,
    **kwargs: Any,
) -> RefinementStep:
    """Run one self-refinement step."""

    # 1. Evaluate current_text.
    previous_evaluation = evaluation_func(current_text)
    previous_score = get_score(previous_evaluation, score_key)

    # 2. Build a refinement prompt using build_refinement_prompt.
    prompt = build_refinement_prompt(
        original_prompt=original_prompt,
        current_text=current_text,
        evaluation=previous_evaluation,
        focus=focus,
        preserve=preserve,
        score_key=score_key,
    )

    # 3. Generate revised text with generation_func.
    revised_text = generation_func(
        prompt,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        **kwargs,
    )

    # 4. Evaluate the revised text.
    revised_evaluation = evaluation_func(revised_text)
    revised_score = get_score(revised_evaluation, score_key)

    # 5. Compare the old score and revised score.
    accepted = revised_score > previous_score

    # 6. Return a RefinementStep.
    #
    # This function should not decide whether to continue the full loop.
    # It only performs one attempt.
    return RefinementStep(
        iteration=1,
        prompt=prompt,
        previous_text=current_text,
        revised_text=revised_text,
        previous_score=previous_score,
        revised_score=revised_score,
        accepted=accepted,
        evaluation=revised_evaluation,
    )


def choose_better_text(
    current_text: str,
    revised_text: str,
    current_evaluation: dict,
    revised_evaluation: dict,
    *,
    score_key: str = "overall_score",
    require_improvement: bool = True,
) -> tuple[str, dict, bool]:
    """Decide whether to accept a revised text."""

    # 1. Compare the current score and revised score.
    current_score = get_score(current_evaluation, score_key)
    revised_score = get_score(revised_evaluation, score_key)

    # 2. If require_improvement is True, accept only strictly higher scores.
    if require_improvement:
        accepted = revised_score > current_score

    # 3. If require_improvement is False, accept equal scores as well.
    else:
        accepted = revised_score >= current_score

    # 4. Return the selected text, selected evaluation, and whether the revision was accepted.
    if accepted:
        return revised_text, revised_evaluation, True
    else:
        return current_text, current_evaluation, False


def should_continue(
    *,
    iteration: int,
    max_iterations: int,
    current_score: float,
    target_score: Optional[float] = None,
    last_improved: bool = True,
    stop_on_no_improvement: bool = True,
) -> bool:
    """Decide whether the refinement loop should continue."""

    # 1. Stop if iteration has reached max_iterations.
    if iteration >= max_iterations:
        return False
    # 2. Stop if target_score is set and current_score is high enough.
    if target_score is not None and current_score >= target_score:
        return False
    # 3. Stop if stop_on_no_improvement is True and last_improved is False.
    if stop_on_no_improvement and not last_improved:
        return False
    
    # 4. Otherwise continue.
    return True

def iterative_refinement(
    original_prompt: str,
    initial_text: str,
    generation_func: GenerationFunction,
    evaluation_func: EvaluationFunction,
    *,
    max_iterations: int = 3,
    target_score: Optional[float] = None,
    temperature: float = 0.7,
    top_p: Optional[float] = 0.9,
    max_new_tokens: int = 160,
    score_key: str = "overall_score",
    stop_on_no_improvement: bool = True,
    require_improvement: bool = True,
    focus: Optional[list[str]] = None,
    preserve: Optional[list[str]] = None,
    **kwargs: Any,
) -> dict:
    """Run the full self-refinement loop."""

    # 1. Evaluate initial_text.
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")

    initial_evaluation = evaluation_func(initial_text)
    initial_score = get_score(initial_evaluation, score_key)

    current_text = initial_text
    current_evaluation = initial_evaluation
    current_score = initial_score

    iteration = 0
    last_improved = True
    iterations = []

    # 2. Repeatedly call refine_once until should_continue says to stop.
    while should_continue(
        iteration=iteration,
        max_iterations=max_iterations,
        current_score=current_score,
        target_score=target_score,
        last_improved=last_improved,
        stop_on_no_improvement=stop_on_no_improvement,
    ):
        step = refine_once(
            original_prompt=original_prompt,
            current_text=current_text,
            generation_func=generation_func,
            evaluation_func=evaluation_func,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            score_key=score_key,
            focus=focus,
            preserve=preserve,
            **kwargs,
        )

    # 3. After each step, keep the better text using choose_better_text.
        previous_score = current_score

        current_text, current_evaluation, accepted = choose_better_text(
            current_text=current_text,
            revised_text=step.revised_text,
            current_evaluation=current_evaluation,
            revised_evaluation=step.evaluation,
            score_key=score_key,
            require_improvement=require_improvement,
        )

        current_score = get_score(current_evaluation, score_key)
        last_improved = current_score > previous_score

    # 4. Track every RefinementStep in a list.
        iteration += 1
        step.iteration = iteration
        step.accepted = accepted
        iterations.append(step)

    # Determine why the refinement loop stopped.
    if iteration >= max_iterations:
        stopped_reason = "max_iterations"
    elif target_score is not None and current_score >= target_score:
        stopped_reason = "target_score"
    elif stop_on_no_improvement and not last_improved:
        stopped_reason = "no_improvement"
    else:
        stopped_reason = "completed"

    # 5. Return a result dictionary with the initial and final information.
    #
    # Be careful to avoid infinite loops.
    return {
        "initial_text": initial_text,
        "initial_score": initial_score,
        "initial_evaluation": initial_evaluation,
        "final_text": current_text,
        "final_score": current_score,
        "final_evaluation": current_evaluation,
        "iterations": iterations,
        "stopped_reason": stopped_reason,
    }


def summarize_refinement(result: dict) -> str:
    """Create a readable summary of a refinement run."""

    # 1. Include the initial score and final score.
    summary = (
        f"Initial score: {result['initial_score']}\n"
        f"Final score: {result['final_score']}\n"
    )

    # 2. Include the number of iterations.
    iterations = result["iterations"]
    summary += f"Number of iterations: {len(iterations)}\n"

    # 3. Include whether each iteration was accepted or rejected.
    for step in iterations:
        status = "Accepted" if step.accepted else "Rejected"
        summary += (
            f"Iteration {step.iteration}: {status}\n"
            f"Previous score: {step.previous_score}\n"
            f"Revised score: {step.revised_score}\n"
        )

    # 4. Include the stopped reason.
    #
    # This is useful for debugging and for showing the tradeoff between extra
    # inference compute and output improvement.
    summary += f"Stopped reason: {result['stopped_reason']}"
    return summary
