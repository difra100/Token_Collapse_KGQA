# Custom graph encoders, projectors, and LLMs

`run_train.py` is the supported public launcher. It registers two models:

- `flex_graph_llm`: a two-layer sigmoid projector;
- `flex_gelu_graph_llm`: the same projector with GELU.

Unlike the upstream `graph_llm`, these models infer the projector output width
from the loaded decoder's input embeddings. This is the supported route for
Llama- and Qwen-style Hugging Face causal language models with different hidden
sizes.

## Choose an LLM

Set `GRAPH_PROMPT_LLM_PATH` to either a Hugging Face identifier or a local model
directory. The model must be loadable by `AutoModelForCausalLM`, expose input
embeddings, and have `q_proj` and `v_proj` modules if `--llm_frozen False` is
used for the included LoRA configuration.

```bash
export GRAPH_PROMPT_LLM_PATH=meta-llama/Llama-2-7b-hf
python run_train.py --dataset webqsp --model_name flex_graph_llm \
  --gnn_model_name gt --llm_model_name llama2 --llm_frozen True
```

For a Qwen checkpoint, change only the path and model label:

```bash
export GRAPH_PROMPT_LLM_PATH=Qwen/Qwen2.5-3B
python run_train.py --dataset webqsp --model_name flex_graph_llm \
  --gnn_model_name gt --llm_model_name qwen25 --llm_frozen False
```

The label supplied through `--llm_model_name` is a local key. It need not be a
predefined upstream name when `GRAPH_PROMPT_LLM_PATH` is set.

## Add a graph encoder

The upstream encoder registry is
`G-Retriever-upstream/src/model/gnn.py:load_gnn_model`. An encoder constructor
receives `in_channels`, `hidden_channels`, `out_channels`, `num_layers`,
`dropout`, and `num_heads`. Its forward method must return `(node_embeddings,
edge_attributes)`, following the included GCN, GAT, and graph transformer.

Add the class to that registry and train with its registry key:

```python
load_gnn_model["my_gnn"] = MyGraphEncoder
```

```bash
python run_train.py --dataset webqsp --model_name flex_graph_llm \
  --gnn_model_name my_gnn --gnn_hidden_dim 1024
```

The graph representation passed to the projector has width
`--gnn_hidden_dim`; set it to the output width of the custom encoder.

## Add a projector

Subclass `FlexibleGraphLLM` in `public_models.py` and override
`build_projector(input_dim, output_dim)`. `input_dim` is the graph encoder
width, and `output_dim` is the loaded LLM embedding width. Register the class
in `register()` and select its key with `--model_name`. This preserves the
upstream loss, prompt construction, optimizer, checkpoint format, and
inference path.

```python
class MyProjector(FlexibleGraphLLM):
    @staticmethod
    def build_projector(input_dim, output_dim):
        return nn.Sequential(nn.Linear(input_dim, output_dim), nn.Tanh())

load_model["my_projector_llm"] = MyProjector
```

## Data adapter contract

The included data adapters live in `G-Retriever-upstream/src/dataset/`. A new
adapter must expose train/validation/test indices and return samples containing
the graph object, `question`, `description`, `label`, and `id`. Reuse the
upstream collator to preserve prompt construction and the graph-token injection
interface.
