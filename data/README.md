# Data directory

The repository includes three raw upstream input files under
`G-Retriever-upstream/dataset/`: the ExplaGraphs TSV, SceneGraphs questions
CSV, and compressed GQA scene-graph archive. Retrieval caches, generated graph
embeddings, and model weights are not distributed.

The included G-Retriever code expects raw inputs and processed datasets under
`G-Retriever-upstream/dataset/`. Run preprocessing from
`G-Retriever-upstream/` (not from its `dataset/` subdirectory).

ExplaGraphs and SceneGraphs preprocessing reads local raw inputs. Extract
`dataset/gqa/sceneGraphs.zip` before processing SceneGraphs. The
[top-level README](../README.md#prepare-data) provides the extraction and
preprocessing commands. Required input paths,
relative to `G-Retriever-upstream/`, are:

| Benchmark | Required raw inputs | Downloaded by preprocessing? |
| --- | --- | --- |
| ExplaGraphs | `dataset/expla_graphs/train_dev.tsv` | No |
| SceneGraphs | `dataset/scene_graphs/questions.csv`, `dataset/gqa/train_sceneGraphs.json` | No |
| WebQSP | Hugging Face dataset `rmanluo/RoG-webqsp` | Yes |

After extracting the scene graphs, run the appropriate preprocessing
commands from the top-level README:

- `python -m src.dataset.preprocess.expla_graphs`
- `python -m src.dataset.preprocess.scene_graphs`
- `python -m src.dataset.preprocess.webqsp`

The built-in loaders do not search this top-level `data/` folder. Keep their
inputs at the paths above. Only the three bundled source files are allowed
through `.gitignore`; extracted JSON files and generated datasets stay local.
This top-level `data/` folder can hold custom datasets or extra downloads.
A custom dataset adapter should return the same sample fields
used by the upstream collator: graph, question, description, label, and id.

Raw inputs are copied unchanged from the
[official G-Retriever dataset directory](https://github.com/XiaoxinHe/G-Retriever/tree/315b0ff8a206536067602fb97e77c10f4d646d5d/dataset).
The original [ExplaGraphs](https://explagraphs.github.io/) and
[GQA](https://cs.stanford.edu/people/dorarad/gqa/) datasets retain their
respective attribution and usage terms.
