from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import ttnn


def _sanitize_key(key: str) -> str:
    return key.replace("/", "__").replace(".", "__")


@dataclass
class TTNNWeightStore:
    root: Path
    mode: str = "load"

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.data: Dict[str, Dict[str, Any]] = {"tensors": {}, "meta": {}}
        if self.manifest_path.exists():
            with self.manifest_path.open("r") as f:
                self.data = json.load(f)

    def save_tensor(self, key: str, tensor: "ttnn.Tensor") -> None:
        file_name = self.root / f"{_sanitize_key(key)}.tensorbin"
        ttnn.dump_tensor(file_name, tensor)
        self.data["tensors"][key] = str(file_name)

    def load_tensor(self, key: str, device=None) -> "ttnn.Tensor":
        path = self.data["tensors"].get(key)
        if path is None:
            raise KeyError(f"Missing tensor key in manifest: {key}")
        return ttnn.load_tensor(path, device=device)

    def save_meta(self, key: str, meta: Dict[str, Any]) -> None:
        self.data["meta"][key] = meta

    def get_meta(self, key: str) -> Optional[Dict[str, Any]]:
        return self.data.get("meta", {}).get(key)

    def flush(self) -> None:
        with self.manifest_path.open("w") as f:
            json.dump(self.data, f, indent=2)


class ModuleKeyRegistry:
    def __init__(self, root_module) -> None:
        self._module_to_name = {id(m): name for name, m in root_module.named_modules()}

    def name_of(self, module) -> Optional[str]:
        return self._module_to_name.get(id(module))
