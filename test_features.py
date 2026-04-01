"""Quick test: verify FEATURES import and auto_join_team state"""
import os
import sys

os.environ["WULFRAM_AUTO_JOIN_TEAM"] = "1"

from wulfram.session import FEATURES
print("before:", FEATURES.auto_join_team)

FEATURES.auto_join_team = True
print("after set:", FEATURES.auto_join_team)

from wulfram.handlers import FEATURES as HF
print("handlers:", HF.auto_join_team, "same:", FEATURES is HF)
print("ids:", id(FEATURES), id(HF))
