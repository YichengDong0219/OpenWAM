"""Durable resource ledger for the recovery pipeline (one GPU allocation at a time).

state.json is replaced atomically (tmp + fsync + rename) and every transition is also appended to events.jsonl.
Protocol per invocation: reserve(full allowed wall) BEFORE any GPU work -> run -> settle(actual monotonic wall).
A restart that finds active_jobs != 0 cannot know whether the leftover allocation was charged/cleaned, so it HALTS
instead of starting another allocation. Nothing is ever reset to zero.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

PHASES = ("V", "Q_eng", "M2", "Q_eval", "M3")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class Ledger:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "state.json"
        self.events = self.root / "events.jsonl"
        if not self.path.exists():
            self._write({"format": "recovery-ledger/1", "created": _now(),
                         "used_card_seconds": {p: 0.0 for p in PHASES},
                         "reserved_card_seconds": {p: 0.0 for p in PHASES},
                         "attempted_invocations": {p: 0 for p in PHASES},
                         "active_jobs": 0, "active": None, "halted": False, "halt_reason": None,
                         "receipts": {}, "validation_attempts": [], "backend": None,
                         "backend_manifest_sha256": None, "invocations": []})
            self._event({"event": "created"})
        st = self.state()
        if st["active_jobs"] != 0 and not st["halted"]:
            st["halted"], st["halt_reason"] = True, f"leftover active allocation found at startup: {st['active']}"
            self._write(st)
            self._event({"event": "halt", "reason": st["halt_reason"]})

    def state(self) -> dict:
        return json.loads(self.path.read_text())

    def _write(self, st: dict) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(st, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def _event(self, row: dict) -> None:
        with open(self.events, "a") as f:
            f.write(json.dumps({"time": _now(), **row}) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def reserve(self, phase: str, seconds: float, inv: dict) -> None:
        st = self.state()
        if st["halted"]:
            raise RuntimeError(f"ledger halted: {st['halt_reason']}")
        if st["active_jobs"] != 0:
            raise RuntimeError("another allocation is active")
        st["reserved_card_seconds"][phase] += float(seconds)
        st["attempted_invocations"][phase] += 1
        st["active_jobs"] = 1
        st["active"] = {"phase": phase, "reserved": float(seconds), "start_utc": _now(),
                        "start_monotonic": time.monotonic(), **inv}
        self._write(st)
        self._event({"event": "reserve", "phase": phase, "seconds": seconds, **inv})

    def settle(self, actual_wall: float, outcome: dict) -> None:
        st = self.state()
        act = st["active"]
        if act is None:
            raise RuntimeError("settle without an active allocation")
        st["reserved_card_seconds"][act["phase"]] -= act["reserved"]
        st["used_card_seconds"][act["phase"]] += float(actual_wall)
        st["active_jobs"], st["active"] = 0, None
        st["invocations"].append({**{k: v for k, v in act.items() if k != "start_monotonic"},
                                  "end_utc": _now(), "actual_wall_s": round(float(actual_wall), 2), **outcome})
        self._write(st)
        self._event({"event": "settle", "phase": act["phase"], "actual_wall_s": actual_wall, **outcome})

    def unresolved(self, reason: str, observed: dict) -> None:
        """No verified end/cleanup evidence: keep the allocation ACTIVE with its full reservation (conservatively charged)
        and halt. Only verified same-token end/cleanup evidence may later settle it; nothing is released here."""
        st = self.state()
        st["halted"], st["halt_reason"] = True, f"unresolved allocation: {reason}"
        st["unresolved_allocation"] = {**(st["active"] or {}), "observed": observed, "time": _now()}
        self._write(st)
        self._event({"event": "unresolved", "reason": reason, "observed": observed})

    def update(self, **fields) -> None:
        st = self.state()
        st.update(fields)
        self._write(st)
        self._event({"event": "update", "fields": sorted(fields)})

    def halt(self, reason: str) -> None:
        st = self.state()
        st["halted"], st["halt_reason"] = True, reason
        self._write(st)
        self._event({"event": "halt", "reason": reason})
