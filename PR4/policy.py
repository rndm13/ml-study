"""Reusable three-zone decision policy. Thresholds come from policy_config.json."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class DecisionPolicy:
    threshold_low: float
    threshold_high: float

    def __post_init__(self):
        if not (math.isfinite(self.threshold_low) and math.isfinite(self.threshold_high)):
            raise ValueError("thresholds must be finite")
        if self.threshold_low > self.threshold_high:
            raise ValueError("threshold_low must be <= threshold_high")

    def decide(self, score: float) -> str:
        if not math.isfinite(float(score)):
            raise ValueError("score must be finite")
        if score >= self.threshold_high:
            return "additional_testing"
        if score >= self.threshold_low:
            return "manual_review"
        return "standard_testing"

    def decide_many(self, scores):
        return [self.decide(float(score)) for score in scores]
