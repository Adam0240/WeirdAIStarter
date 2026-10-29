"""
Best-of-N generation utilities for Weird AI.

This module connects the Chapter 4 idea of inference-time scaling to the
Weird AI project.

The core idea is:

    generate one candidate     -> maybe good, maybe not
    generate several candidates -> evaluate each one
    select the best candidate   -> better final output without retraining

Students complete this file as the applied assignment for this lesson.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional


GenerationFunction = Callable[..., str]
EvaluationFunction = Callable[[str], dict]


@dataclass
class GenerationCandidate:
    """
    Store one generated candidate and its evaluation results.

    Attributes:
        text: The generated text.
        score: Overall evaluation score for the text.
        evaluation: Full evaluation dictionary returned by the evaluator.
        index: Candidate number, starting at 1.
    """

    text: str
    score: float
    evaluation: dict
    index: int


def build_generation_prompt(topic: str, style: str = "emo parody", line_count: int = 8) -> str:
    """
    Build a prompt for Weird AI parody generation.

    Args:
        topic: The subject of the parody.
        style: The desired lyric style.
        line_count: Target number of lyric lines.

    Returns:
        A formatted prompt string.
    """

    # 1. Create a clear prompt asking the model to write parody lyrics.
    prompt = (
        "Write parody lyrics."

    # 2. Include the topic, style, and target line count.
        f" The topic is {topic}, the style is {style}, and the song should have {line_count} lines."

    # 3. Ask the model to return only the lyrics, not an explanation.
        " Return only the lyrics without any explanations."
    )

    return prompt

def generate_once(
    prompt: str,
    generation_func: GenerationFunction,
    *,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    max_new_tokens: int = 120,
    **kwargs: Any,
) -> str:
    """
    Generate a single candidate response.

    This function is intentionally model-agnostic. The caller provides
    generation_func so this utility can work with a real model, a test double,
    or a future API wrapper.

    Args:
        prompt: Prompt text.
        generation_func: Callable that generates text.
        temperature: Sampling temperature.
        top_p: Optional nucleus-sampling cutoff.
        max_new_tokens: Maximum number of tokens to generate.
        **kwargs: Additional arguments forwarded to generation_func.

    Returns:
        Generated text.
    """


    # 1. Call generation_func.
    generated_text = generation_func(

    # 2. Pass prompt, temperature, top_p, max_new_tokens, and any extra kwargs.
    prompt,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        **kwargs
    )

    # 3. Return the generated text as a string.
    return str(generated_text)


def generate_multiple(
    prompt: str,
    generation_func: GenerationFunction,
    *,
    num_candidates: int = 5,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    max_new_tokens: int = 120,
    **kwargs: Any,
) -> list[str]:
    """
    Generate several candidate responses for the same prompt.

    This is the Weird AI version of inference-time scaling: spend more
    generation compute to get multiple possible outputs.

    Args:
        prompt: Prompt text.
        generation_func: Callable that generates text.
        num_candidates: Number of candidates to generate.
        temperature: Sampling temperature.
        top_p: Optional nucleus-sampling cutoff.
        max_new_tokens: Maximum number of tokens to generate.
        **kwargs: Additional arguments forwarded to generation_func.

    Returns:
        A list of generated strings.
    """

    # TODO:
    # 1. Validate that num_candidates is at least 1.
    if num_candidates < 1:
        raise ValueError("num_candidates must be at least 1.")
    
    # 2. Call generate_once num_candidates times.
    candidates = []
    for i in range(num_candidates):
        generated_text = generate_once(
            prompt,
            generation_func,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        candidates.append(generated_text)
   
    # 3. Return all generated candidates in a list.
    return candidates

def score_candidate(
    text: str,
    evaluation_func: EvaluationFunction,
    *,
    score_key: str = "overall_score",
) -> GenerationCandidate:
    """
    Evaluate a single generated candidate.

    Args:
        text: Candidate text.
        evaluation_func: Function that returns an evaluation dictionary.
        score_key: Dictionary key containing the main score.

    Returns:
        A GenerationCandidate with index set to 0. The caller may replace it.
    """

    # 1. Call evaluation_func(text).
    evaluation = evaluation_func(text)
    # 2. Read the score from score_key.   
    # 3. If score_key is missing, use 0.0 as the score.
    score = float(evaluation.get(score_key, 0.0))
    # 4. Return a GenerationCandidate.
    return GenerationCandidate(
        text=text,
        score=score,
        evaluation=evaluation,
        index=0
    )

def rank_candidates(
    texts: Iterable[str],
    evaluation_func: EvaluationFunction,
    *,
    score_key: str = "overall_score",
) -> list[GenerationCandidate]:
    """
    Evaluate and rank generated candidates from best to worst.

    Args:
        texts: Generated candidate texts.
        evaluation_func: Function that evaluates one candidate.
        score_key: Dictionary key containing the main score.

    Returns:
        GenerationCandidate objects sorted by descending score.
    """

    # 1. Evaluate each text using score_candidate.
    candidates = []
    for index, text in enumerate(texts, start=1):
        candidate = score_candidate(
            text,
            evaluation_func,
            score_key=score_key
        )
        # 2. Store candidate numbers starting at 1.
        candidate.index = index
        candidates.append(candidate)

    # 3. Sort candidates so the highest score comes first.
    ranked_candidates = sorted(
        candidates,
        key=lambda c: c.score,
        reverse=True
    )
    # 4. Return the sorted list.
    return ranked_candidates


def select_best_candidate(candidates: list[GenerationCandidate]) -> GenerationCandidate:
    """
    Select the best candidate from a ranked or unranked list.

    Args:
        candidates: Candidate objects.

    Returns:
        The candidate with the highest score.
    """

    # 1. Validate that the candidates list is not empty.
    if not candidates:
        raise ValueError("The candidates list cannot be empty.")
    
    # 2. Return the candidate with the highest score.
    return max(candidates, key=lambda c: c.score)


def generate_best(
    prompt: str,
    generation_func: GenerationFunction,
    evaluation_func: EvaluationFunction,
    *,
    num_candidates: int = 5,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    max_new_tokens: int = 120,
    score_key: str = "overall_score",
    **kwargs: Any,
) -> dict:
    """
    Generate multiple candidates, evaluate them, and return the best result.

    Args:
        prompt: Prompt text.
        generation_func: Callable that generates text.
        evaluation_func: Callable that evaluates text.
        num_candidates: Number of candidates to generate.
        temperature: Sampling temperature.
        top_p: Optional nucleus-sampling cutoff.
        max_new_tokens: Maximum number of tokens to generate.
        score_key: Dictionary key containing the main score.
        **kwargs: Additional arguments forwarded to generation_func.

    Returns:
        A dictionary containing:
            - best_text
            - best_score
            - best_candidate
            - candidates
    """

    # 1. Generate multiple text candidates.
    texts = generate_multiple(
        prompt,
        generation_func,
        num_candidates=num_candidates,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        **kwargs
    )
    # 2. Rank the candidates using evaluation_func.
    ranked_candidates = rank_candidates(
        texts,
        evaluation_func,
        score_key=score_key
    )
    # 3. Select the best candidate.
    best_candidate = select_best_candidate(ranked_candidates)

    # 4. Return a dictionary containing the best text, score, candidate, and all ranked candidates.
    return {
        "best_text": best_candidate.text,
        "best_score": best_candidate.score,
        "best_candidate": best_candidate,
        "candidates": ranked_candidates
    }

def format_candidate_report(candidates: list[GenerationCandidate]) -> str:
    """
    Format candidate scores as a readable report.

    Args:
        candidates: Candidate objects.

    Returns:
        A multi-line report string.
    """

    # 1. Create one report section per candidate.
    sections = []

    for candidate in candidates:
        section = []

        # 2. Include candidate index and overall score.
        section.append(f"Candidate {candidate.index}")
        section.append(f"Overall Score: {candidate.score}")

        # 3. Include rhyme, syllable, or structure scores when available.
        for key in ["rhyme_score", "syllable_consistency_score", "structure_score"]:
            if key in candidate.evaluation:
                section.append(f"{key}: {candidate.evaluation[key]}")

        sections.append("\n".join(section))

    # 4. Return the sections joined with blank lines.
    return "\n\n".join(sections)

