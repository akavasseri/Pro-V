"""Behavioral reference model (FRM) for a 2-input AND gate.

This is the *source of truth* the functional coverage agent reads. The agent
never looks inside top_module.v to decide what to test; it drives THIS model.
"""

import json
from typing import Dict


class GoldenDUT:
    def load(self, inputs: Dict[str, str]) -> Dict[str, str]:
        a = int(inputs["a"], 2)
        b = int(inputs["b"], 2)
        return {"y": str(a & b)}
