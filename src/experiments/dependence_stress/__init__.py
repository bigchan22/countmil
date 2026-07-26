"""Conditional-independence stress-test utilities for rebuttal experiments."""

from .dgp import DependenceData, generate_dependence_data, within_bag_label_correlation

__all__ = ["DependenceData", "generate_dependence_data", "within_bag_label_correlation"]
