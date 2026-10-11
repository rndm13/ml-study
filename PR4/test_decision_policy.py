import unittest
from policy import DecisionPolicy

class DecisionPolicyTests(unittest.TestCase):
    def setUp(self): self.policy = DecisionPolicy(0.30, 0.70)
    def test_below_low(self): self.assertEqual(self.policy.decide(0.29), "standard_testing")
    def test_equal_low(self): self.assertEqual(self.policy.decide(0.30), "manual_review")
    def test_between_thresholds(self): self.assertEqual(self.policy.decide(0.50), "manual_review")
    def test_equal_high(self): self.assertEqual(self.policy.decide(0.70), "additional_testing")
    def test_above_high(self): self.assertEqual(self.policy.decide(0.91), "additional_testing")
    def test_invalid_scores(self):
        for score in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(score=score), self.assertRaises(ValueError): self.policy.decide(score)
    def test_reversed_thresholds(self):
        with self.assertRaises(ValueError): DecisionPolicy(0.8, 0.2)
    def test_non_finite_threshold(self):
        with self.assertRaises(ValueError): DecisionPolicy(float("nan"), 0.5)
    def test_negative_logit_scores_are_valid(self):
        policy = DecisionPolicy(-2.0, 3.0)
        self.assertEqual(policy.decide(-3.0), "standard_testing")
        self.assertEqual(policy.decide(4.0), "additional_testing")
    def test_batch_order_preserved(self):
        self.assertEqual(self.policy.decide_many([0.8, 0.2, 0.5]),
                         ["additional_testing", "standard_testing", "manual_review"])
    def test_equal_thresholds_high_action_wins(self):
        self.assertEqual(DecisionPolicy(0.5, 0.5).decide(0.5), "additional_testing")

if __name__ == "__main__": unittest.main(verbosity=2)
