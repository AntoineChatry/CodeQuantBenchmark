"""Story 3.3 : Métriques de benchmark."""

import math


def jaccard_similarity(text_a: str, text_b: str) -> float:
    tokens_a = set(text_a.split())
    tokens_b = set(text_b.split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def syntax_validity(code: str) -> bool:
    try:
        compile(code, "<benchmark>", "exec")
        return True
    except (SyntaxError, ValueError):  # ValueError = null bytes
        return False


def extract_code_blocks(text: str) -> str:
    blocks: list[str] = []
    in_block = False
    current: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("```"):
            if in_block:
                blocks.append("\n".join(current))
                current = []
                in_block = False
            else:
                in_block = True
        elif in_block:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return "\n\n".join(blocks) if blocks else text


def bleu_score(reference: str, candidate: str, max_n: int = 4) -> float:
    """BLEU score (Papineni et al., 2002).

    BLEU = BP * exp( sum_{i=1}^{N} w_i * ln(p_i) )

    BP = 1                    if c >= r
         exp(1 - r/c)         if c < r

    p_i = modified n-gram precision (clipped by max ref count)
    w_i = 1/N (uniform weights)
    c = candidate length, r = reference length
    """
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    c = len(cand_tokens)
    r = len(ref_tokens)

    if c == 0 or r == 0:
        return 0.0

    # Brevity Penalty
    if c >= r:
        bp = 1.0
    else:
        bp = math.exp(1 - r / c)

    # Modified n-gram precisions
    precisions: list[float] = []
    for n in range(1, max_n + 1):
        ref_ngrams: dict[tuple[str, ...], int] = {}
        for i in range(r - n + 1):
            ng = tuple(ref_tokens[i:i + n])
            ref_ngrams[ng] = ref_ngrams.get(ng, 0) + 1

        cand_ngrams: dict[tuple[str, ...], int] = {}
        for i in range(c - n + 1):
            ng = tuple(cand_tokens[i:i + n])
            cand_ngrams[ng] = cand_ngrams.get(ng, 0) + 1

        if not cand_ngrams:
            precisions.append(0.0)
            continue

        clipped = sum(
            min(count, ref_ngrams.get(ng, 0))
            for ng, count in cand_ngrams.items()
        )
        total = sum(cand_ngrams.values())
        precisions.append(clipped / total if total > 0 else 0.0)

    # If any precision is 0, BLEU = 0 (ln(0) undefined)
    if any(p == 0.0 for p in precisions):
        return 0.0

    # Weighted average of log precisions (uniform weights w_i = 1/N)
    w = 1.0 / len(precisions)
    log_avg = sum(w * math.log(p) for p in precisions)
    return round(bp * math.exp(log_avg), 4)


def ms_per_token(latency_ms: float, num_tokens: int) -> float:
    if num_tokens == 0:
        return 0.0
    return round(latency_ms / num_tokens, 2)
