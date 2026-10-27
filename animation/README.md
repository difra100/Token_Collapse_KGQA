# Token collapse animation (Manim)

An explainer video for the paper, about 2 min 45 s long, in 7 scenes:

| Scene | Content |
|---|---|
| `Title` | Title and authors; many tokens shrink into one point |
| `Pipeline` | G-Retriever: graph → encoder → Sigmoid projector → graph token, plus the verbalised graph in the prompt; the paper's summary (Sec. 1) of how G-Retriever interprets its ablations |
| `Collapse` | Encoder outputs are spread out (cos 0.593); after the projector every token sits on the shared mean μ (cos 0.996, residual ratio 0.057), with a zoom inset |
| `Interventions` | WebQSP REAL / MEAN / SWAP / GSWAP / CONST / NOTXT bars, then the highlighted **soft prompt** verdict |
| `Beyond` | Appendix F audit: GRAG-PCST, GRETPool (TopK, SAG, DiffPool K=8/100) and GraSP-PCST all collapse (cosine 0.998–1.000, Table 10); token interventions unresolved with text, −25.6 to −43.2 points without text; single-seed caveat |
| `TwoStage` | Stage 1 InfoNCE alignment + freeze, Stage 2 graph token + prompt token, then the token clouds under the same title |
| `Takeaway` | The five diagnoses of Sec. 4.5 (SOFT PROMPT, PRIOR, DECODABLE, MASKED, USED) with Table 5 examples, and the closing message |

Every number shown on screen is taken from the paper. The point clouds are schematic
(synthetic) and labelled as such on screen; they are scaled to the paper's reported
statistics (Figure 1 mean radii 0.03 and 0.71), not plotted from stored tokens.

Pacing: `SLOW = 1.25` near the top of the script stretches every animation and pause.

## Setup

Manim needs cairo, pango and ffmpeg. The simplest way to get them is conda-forge
(LaTeX is **not** needed, as the script only uses `Text`):

```bash
micromamba create -p ./.venv -c conda-forge python=3.11 manim
```

## Render

```bash
# quick preview of one scene (480p15)
.venv/bin/manim -ql token_collapse.py Collapse

# full video, 1080p60
.venv/bin/manim -qh token_collapse.py FullVideo

# 4K for slides
.venv/bin/manim -qk token_collapse.py FullVideo
```

The videos are written to `media/videos/token_collapse/<quality>/`.
Colours, the font and helper functions are at the top of `token_collapse.py`.
Each scene is an `act_*` function, so you can reorder or drop scenes in `FullVideo`.

## Teaser cut (social media)

`Teaser` is a ~1 min 30 s cut: `Title`, `Pipeline`, `Collapse`, then the first page of `Beyond` (the Appendix F audit:
GRAG-PCST, GRETPool TopK / SAG / DiffPool K=8 / K=100, and GraSP-PCST all collapse,
inter-sample cosine 0.998–1.000, Table 10), and a closing screen with the two open questions
pointing to the paper and the full video.

```bash
.venv/bin/manim -qh token_collapse.py Teaser
```

The `Causes` scene (projector × training analysis) is no longer part of `FullVideo`,
but it can still be rendered on its own.
