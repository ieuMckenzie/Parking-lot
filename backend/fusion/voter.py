from collections import defaultdict

from backend.fusion.models import FusionResult, FusionStatus, Read


def vote(
    reads: list[Read],
    min_reads: int = 3,
    min_confidence: float = 0.6,
) -> list[FusionResult]:
    """Confidence-weighted majority vote across reads, grouped by class_name."""
    if not reads:
        return []

    by_class: dict[str, list[Read]] = defaultdict(list)
    for r in reads:
        by_class[r.class_name].append(r)

    results: list[FusionResult] = []
    for class_name, class_reads in by_class.items():
        scores: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for r in class_reads:
            scores[r.text] += r.confidence
            counts[r.text] += 1

        winner_text = max(scores, key=scores.get)  # type: ignore[arg-type]
        winner_score = scores[winner_text]
        winner_count = counts[winner_text]
        total_reads = len(class_reads)

        if winner_count >= min_reads and winner_score >= min_confidence:
            status = FusionStatus.CONFIRMED
        else:
            status = FusionStatus.NEEDS_REVIEW

        results.append(FusionResult(
            class_name=class_name,
            value=winner_text,
            confidence=winner_score,
            num_reads=winner_count,
            consensus_ratio=winner_count / total_reads,
            status=status,
        ))

    return results
