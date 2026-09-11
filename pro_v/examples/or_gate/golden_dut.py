"""Behavioral reference model (FRM) for a 2-input OR gate.

Paired with the AND example to make the point concrete: the tests 00 and 11
give the SAME outputs for AND and OR, so they cannot tell the two apart. Only
the mixed cases 01 and 10 -- which the agent selects -- distinguish them.
"""

import json
from typing import Dict


class GoldenDUT:
    def load(self, inputs: Dict[str, str]) -> Dict[str, str]:
        a = int(inputs["a"], 2)
        b = int(inputs["b"], 2)
        return {"y": str(a | b)}
