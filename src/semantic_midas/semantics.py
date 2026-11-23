"""Strict offline interface for an external language-to-direction compiler.

No network calls and no claim that synthetic sign-channel tests measure LLM
accuracy. The demo parser accepts ONLY a controlled grammar; it is not a
real announcement understanding system.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

PRIMITIVES = ("activity_impact", "cost_recency", "financial_transmission")
ALLOWED = {"announcement_id", "as_of", "directions", "evidence", "model", "prompt_sha256"}


def validate_cached_prior(record: dict, q: int = 3) -> list[int]:
    unknown = set(record) - ALLOWED
    if unknown:
        raise ValueError(f"Forbidden output fields (including magnitudes): {sorted(unknown)}")
    if not {"announcement_id", "as_of", "directions"} <= set(record):
        raise ValueError("Missing required announcement metadata")
    if q not in (1,3):
        raise ValueError("q must be 1 or 3")
    names = PRIMITIVES[:q]
    directions = record["directions"]
    if not isinstance(directions, dict) or set(directions) != set(names):
        raise ValueError("Directions must contain exactly the frozen named coordinates")
    a=[]
    for key in names:
        val=directions[key]
        if isinstance(val,bool) or not isinstance(val,int) or val not in (-1,0,1):
            raise ValueError("Directions must be integer -1,0,+1; no confidence/magnitude")
        a.append(val)
    return a


def read_cache(path: str | Path, q: int = 3) -> list[dict]:
    records=[]; ids=set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r=json.loads(line); validate_cached_prior(r,q)
        if r["announcement_id"] in ids:
            raise ValueError("Duplicate cached announcement_id")
        ids.add(r["announcement_id"]); records.append(r)
    return records


def controlled_grammar_parser(text: str, q: int = 3) -> dict[str,int]:
    out={}
    for name in PRIMITIVES[:q]:
        hits=re.findall(rf"\b{re.escape(name)}\s+(increases|decreases|unclear)\b",text.lower())
        if len(hits)>1:
            raise ValueError("Ambiguous/repeated controlled-grammar coordinate")
        out[name]={"increases":1,"decreases":-1,"unclear":0}.get(hits[0] if hits else "unclear",0)
    return out
