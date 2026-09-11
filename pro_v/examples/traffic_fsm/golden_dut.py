"""Behavioral reference model (FRM) for a small request/grant FSM.

States: 0=IDLE, 1=REQ, 2=BUSY, 3=DONE.
  IDLE  --start=1--> REQ      (else stays IDLE)
  REQ   ---------->  BUSY
  BUSY  --stall=0--> DONE     (stall=1 holds in BUSY)
  DONE  ---------->  IDLE
  any   --rst=1--->  IDLE

Sequential contract: state lives in self; load(clk, inputs) advances on clk==1.
A fresh GoldenDUT() is the reset state. The coverage agent uses BFS over THIS
model to find the shortest clocked input sequence that reaches each state
(e.g. DONE) and to exercise the transitions -- it never inspects the RTL.
"""

import json
from typing import Dict

IDLE, REQ, BUSY, DONE = 0, 1, 2, 3


class GoldenDUT:
    def __init__(self):
        self.state = IDLE

    def load(self, clk: int, inputs: Dict[str, str]) -> Dict[str, str]:
        start = int(inputs.get("start", "0"), 2)
        stall = int(inputs.get("stall", "0"), 2)
        rst = int(inputs.get("rst", "0"), 2)
        if clk == 1:
            if rst:
                self.state = IDLE
            elif self.state == IDLE:
                self.state = REQ if start else IDLE
            elif self.state == REQ:
                self.state = BUSY
            elif self.state == BUSY:
                self.state = BUSY if stall else DONE
            elif self.state == DONE:
                self.state = IDLE
        return {
            "grant": "1" if self.state == BUSY else "0",
            "done": "1" if self.state == DONE else "0",
        }
