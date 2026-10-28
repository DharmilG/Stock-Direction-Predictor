from __future__ import annotations

"""Phase 6/7 adapter export helper.

CODE READY — UNTESTED
"""

import argparse
import shutil
import subprocess
from pathlib import Path

from ..qpipeline.contracts import load_q_track_config


def _write_modelfile(path: Path, base_model: str, adapter_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"FROM {base_model}\nADAPTER {adapter_path.as_posix()}\n", encoding="utf-8"
    )


def export_adapter_to_ollama(
    adapter_dir: str | Path,
    model_name: str = "",
    base_model: str = "qwen3:8b",
    ollama_bin: str = "ollama",
) -> dict:
    """Create an Ollama model from the trained LoRA adapter."""
    cfg = load_q_track_config().get("models", {})
    model_name = model_name or cfg.get("lora_model", "qwen3-8b-lora-r1")
    base_model = base_model or cfg.get("base_model", "qwen3:8b")
    adapter_dir = Path(adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter dir not found: {adapter_dir}")
    adapter_gguf = adapter_dir / "adapter_model.gguf"
    if not adapter_gguf.exists():
        # Fallback: assume safetensors adapter is already GGUF-compatible through
        # unsloth's save_pretrained_gguf path; otherwise this is a manual step.
        candidates = list(adapter_dir.glob("*.gguf"))
        if not candidates:
            raise FileNotFoundError(
                "No .gguf adapter found. Run unsloth save_pretrained_gguf first."
            )
        adapter_gguf = candidates[0]
    modelfile = adapter_dir / "Modelfile.qwen_lora"
    _write_modelfile(modelfile, base_model, adapter_gguf)
    exe = shutil.which(ollama_bin) or ollama_bin
    result = subprocess.run(
        [exe, "create", model_name, "-f", str(modelfile)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    return {
        "model_name": model_name,
        "base_model": base_model,
        "adapter_path": str(adapter_gguf),
        "modelfile": str(modelfile),
        "returncode": result.returncode,
        "stdout": (result.stdout or "").strip(),
        "stderr": (result.stderr or "").strip(),
        "pass": result.returncode == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Phase 6 adapter to Ollama.")
    parser.add_argument("--adapter-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--base-model", type=str, default="")
    args = parser.parse_args()
    print(
        export_adapter_to_ollama(
            adapter_dir=args.adapter_dir,
            model_name=args.model_name,
            base_model=args.base_model,
        )
    )


if __name__ == "__main__":
    main()
