"""
Profile engine for WaterSIC model quantization.
Matches tensor names against rules defined in a profile JSON file.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, Tuple, Optional


class QuantProfile:
    def __init__(self, profile_data: Dict[str, Any]):
        self.name = profile_data.get("profile_name", "unnamed_profile")
        self.model_type = profile_data.get("model_type", "unknown")
        self.description = profile_data.get("description", "")
        self.default_action = profile_data.get("default_action", "keep_original")
        self.default_bits = profile_data.get("default_bits", 0)
        
        self.rules = []
        for r in profile_data.get("rules", []):
            pattern = re.compile(r["pattern"], re.IGNORECASE)
            self.rules.append({
                "pattern": pattern,
                "raw_pattern": r["pattern"],
                "action": r.get("action", "keep_original"),
                "target_bits": r.get("target_bits", 0),
                "dtype": r.get("dtype", "original"),
                "description": r.get("description", "")
            })

    @classmethod
    def load_from_file(cls, filepath: str | Path) -> "QuantProfile":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    def match_tensor(self, tensor_name: str) -> Dict[str, Any]:
        """
        Match a tensor name against profile rules.
        Returns rule dict with action, target_bits, dtype, and description.
        """
        for r in self.rules:
            if r["pattern"].search(tensor_name):
                return {
                    "matched_pattern": r["raw_pattern"],
                    "action": r["action"],
                    "target_bits": r["target_bits"],
                    "dtype": r["dtype"],
                    "description": r["description"]
                }
        
        return {
            "matched_pattern": "default",
            "action": self.default_action,
            "target_bits": self.default_bits,
            "dtype": "original" if self.default_bits == 0 else f"watersic_{self.default_bits}b",
            "description": "Default rule"
        }
