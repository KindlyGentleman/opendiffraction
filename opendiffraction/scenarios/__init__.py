"""Experimental acquisition scenarios module."""

from .presets import (
    SCENARIOS,
    ScenarioSpec,
    sample_scenario_config,
    get_scenario_for_profile,
    get_scenario_physics,
)

__all__ = [
    "SCENARIOS",
    "ScenarioSpec",
    "sample_scenario_config",
    "get_scenario_for_profile",
    "get_scenario_physics",
]
