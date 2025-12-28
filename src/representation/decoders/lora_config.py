from dataclasses import dataclass
from typing import List


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: List[str] | None = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"