"""Portable entrypoint for training a graph encoder, projector, and causal LLM.

Run from the repository root.  Set GRAPH_PROMPT_LLM_PATH to a Hugging Face
model identifier or local checkpoint directory.  All remaining arguments are
the upstream G-Retriever training arguments.
"""

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "G-Retriever-upstream"
if not UPSTREAM.is_dir():
    raise FileNotFoundError(f"Missing upstream source tree: {UPSTREAM}")

os.chdir(UPSTREAM)
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(ROOT))

import controls  # noqa: E402
import public_models  # noqa: E402
import train  # noqa: E402
from src.config import parse_args_llama  # noqa: E402
from src.model import llama_model_path  # noqa: E402


controls.register()
public_models.register()
args = parse_args_llama()

llm_path = os.environ.get("GRAPH_PROMPT_LLM_PATH")
if llm_path:
    llama_model_path[args.llm_model_name] = llm_path
elif args.llm_model_name not in llama_model_path:
    raise ValueError(
        "Set GRAPH_PROMPT_LLM_PATH when using a custom --llm_model_name."
    )

if args.output_dir == "output":
    args.output_dir = str(ROOT / "outputs")

print(
    f"[public-train] dataset={args.dataset} model={args.model_name} "
    f"gnn={args.gnn_model_name} llm={args.llm_model_name} "
    f"output_dir={args.output_dir}",
    flush=True,
)
train.main(args)
