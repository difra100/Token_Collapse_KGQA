"""Stage 0 (Paper B): self-supervised alignment pretraining of G-Retriever's graph encoder.

NO LLM is loaded. Only the graph encoder (and, optionally, a question-conditioning layer)
is trained, with an InfoNCE objective that pulls the mean-pooled graph representation
towards a frozen sentence embedding of the SAME verbalised subgraph the LLM would read
(`samples['desc']`, src/dataset/expla_graphs.py:30), and away from other graphs'.

Why the encoder output and not the graph token: PAPER_SKELETON 5.5 measured that a frozen
pretrained encoder with only the projector trained gives cos .846 / CV .54 / eff-rank 75.7
and significant shuffle and random-node effects, while the same encoder fine-tuned by the
QA loss degrades to cos .939. So the encoder is pretrained here and frozen in stage 2.

Question conditioning (ALIGN_QCOND=1, the default):
    x = graphs.x + W_q(sbert(question))[graphs.batch]
W_q is ZERO-INITIALISED, so at step 0 this is bit-identical to upstream G-Retriever and
the no-conditioning ablation is exactly W_q = 0. The question is encoded with the repo's
own all-roberta-large-v1, the model that produced graphs.x, so the two live in one space.

Metrics, on training batches and on the full validation split, appended to
logs/stage0_<run>.jsonl: inter-sample cosine, CV, residual ratio, effective rank (the same
four the probe reports, same formulas) and InfoNCE top-1 retrieval accuracy.

  python stage0_align.py --dataset expla_graphs --gnn_model_name gt --seed 0 --num_epochs 20
Env: ALIGN_TEMP (0.07) ALIGN_BANK (2048) ALIGN_EVERY (50) ALIGN_QCOND (1) ALIGN_OUT (stage0)
All other arguments are the repo's own (src/config.py).
"""
import json
import math
import os
import sys
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
UP = os.environ.get("GRAPH_PROMPT_UPSTREAM", os.path.join(HERE, "G-Retriever-upstream"))
os.chdir(UP)
sys.path.insert(0, UP)
sys.path.insert(0, HERE)

import torch                                        # noqa: E402
import torch.nn as nn                               # noqa: E402
import torch.nn.functional as F                     # noqa: E402
from torch.utils.data import DataLoader             # noqa: E402
from torch_scatter import scatter                   # noqa: E402

from src.config import parse_args_llama             # noqa: E402
from src.dataset import load_dataset                # noqa: E402
from src.model.gnn import load_gnn_model            # noqa: E402
from src.utils.collate import collate_fn            # noqa: E402
from src.utils.lm_modeling import load_sbert, sber_text2embedding   # noqa: E402
from src.utils.seed import seed_everything          # noqa: E402

from collapse_metrics import stats, top1_retrieval   # noqa: E402  shared with train_control/probe
import xfit                                              # noqa: E402  cross-fitting split

TEMP = float(os.environ.get("ALIGN_TEMP", 0.07))
BANK = int(os.environ.get("ALIGN_BANK", 2048))
EVERY = int(os.environ.get("ALIGN_EVERY", 50))
QCOND = os.environ.get("ALIGN_QCOND", "1") != "0"
# What the graph representation is aligned TO.
#   desc        the verbalised subgraph (~15 tokens on ExplaGraphs; 300-900 on SceneGraphs,
#               where all-roberta-large-v1 truncates at 512)
#   full_answer the full answer sentence ("Yes, the banana is on the table"). Task-relevant,
#               and unlike the bare label it is distinct per question, so the 698/2000
#               yes/no items are not identical targets that poison the InfoNCE denominator.
#   answer      the bare label; degenerate wherever the answer space is small.
TARGET = os.environ.get("ALIGN_TARGET", "desc")
TARGET_FIELD = {"desc": "desc", "full_answer": "full_label", "answer": "label"}[TARGET]
OUTDIR = os.path.join(HERE, os.environ.get("ALIGN_OUT", "stage0"))

args = parse_args_llama()
seed_everything(args.seed)
dev = torch.device("cuda")
NOGRAPH = os.environ.get("ALIGN_NOGRAPH", "0") == "1"   # zero the node features: question only
RUN = (f"{args.dataset}_{args.gnn_model_name}_seed{args.seed}"
       + ("" if TARGET == "desc" else f"_{TARGET}") + ("" if QCOND else "_noqcond")
       + ("_nograph" if NOGRAPH else "")
       + (f"_half{xfit.which()}" if xfit.which() else ""))
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
LOG = os.path.join(HERE, "logs", f"stage0_{RUN}.jsonl")
CKPT = os.path.join(OUTDIR, f"{RUN}.pt")
# never clobber a finished run (a smoke test once overwrote a 30-epoch checkpoint with a
# 1-epoch one). ALIGN_TAG namespaces a variant; ALIGN_FORCE=1 overwrites deliberately.
if os.environ.get("ALIGN_TAG"):
    RUN = f"{RUN}_{os.environ['ALIGN_TAG']}"
    LOG = os.path.join(HERE, "logs", f"stage0_{RUN}.jsonl")
    CKPT = os.path.join(OUTDIR, f"{RUN}.pt")
if os.path.exists(CKPT) and os.environ.get("ALIGN_FORCE", "0") != "1":
    sys.exit(f"{CKPT} exists. Set ALIGN_TAG=<name> for a variant, or ALIGN_FORCE=1 to overwrite.")
open(LOG, "w").close()


# ------------------------------------------------------------------ model (no LLM)
# built exactly as GraphLLM builds it (src/model/graph_llm.py:77-84)
encoder = load_gnn_model[args.gnn_model_name](
    in_channels=args.gnn_in_dim,
    out_channels=args.gnn_hidden_dim,
    hidden_channels=args.gnn_hidden_dim,
    num_layers=args.gnn_num_layers,
    dropout=args.gnn_dropout,
    num_heads=args.gnn_num_heads,
).to(dev)

W_q = nn.Linear(args.gnn_in_dim, args.gnn_in_dim).to(dev)
nn.init.zeros_(W_q.weight)                  # at init: identical to upstream G-Retriever
nn.init.zeros_(W_q.bias)

params = list(encoder.parameters()) + (list(W_q.parameters()) if QCOND else [])
n_train = sum(p.numel() for p in params)
print(f"[stage0] run={RUN} trainable={n_train:,} (encoder {sum(p.numel() for p in encoder.parameters()):,}"
      f"{' + W_q ' + format(sum(p.numel() for p in W_q.parameters()), ',') if QCOND else ''})  "
      f"temp={TEMP} bank={BANK} qcond={QCOND} target={TARGET} nograph={NOGRAPH}")
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

# frozen sentence encoder: the SAME model that produced graphs.x
sb_model, sb_tok, sb_dev = load_sbert()
for p in sb_model.parameters():
    p.requires_grad_(False)
sb_model.eval()
_TXT_CACHE = {}


def sbert(texts, keys=None):
    """Frozen all-roberta-large-v1 embeddings, cached by key when one is given."""
    if keys is None:
        return sber_text2embedding(sb_model, sb_tok, sb_dev, list(texts)).to(dev).float()
    todo = [(k, t) for k, t in zip(keys, texts) if k not in _TXT_CACHE]
    if todo:
        emb = sber_text2embedding(sb_model, sb_tok, sb_dev, [t for _, t in todo])
        for (k, _), e in zip(todo, emb):
            _TXT_CACHE[k] = e
    return torch.stack([_TXT_CACHE[k] for k in keys]).to(dev).float()


def target_text(batch):
    """The strings this run aligns to; keys are cached so each is embedded once."""
    if TARGET_FIELD not in batch:
        raise KeyError(f"{args.dataset} has no '{TARGET_FIELD}' field (ALIGN_TARGET={TARGET}); "
                       f"available: {sorted(batch)}")
    return [str(t) for t in batch[TARGET_FIELD]], [(TARGET, i) for i in batch["id"]]


def encode(batch):
    """Mean-pooled graph representation, optionally conditioned on the question."""
    g = batch["graph"].to(dev)
    x = torch.zeros_like(g.x) if NOGRAPH else g.x
    if QCOND:
        q = sbert(batch["question"], keys=[("q", i) for i in batch["id"]])
        x = x + W_q(q)[g.batch]
    n_embeds, _ = encoder(x, g.edge_index.long(), g.edge_attr)
    return scatter(n_embeds, g.batch, dim=0, reduce="mean")


# ------------------------------------------------------------------ metrics
def infonce(z, t, bank_t=None):
    """z: graph reps, t: their targets. Positives are the diagonal; bank adds negatives."""
    zn, tn = F.normalize(z, dim=-1), F.normalize(t, dim=-1)
    all_t = tn if bank_t is None or bank_t.numel() == 0 else torch.cat([tn, F.normalize(bank_t, dim=-1)])
    logits = zn @ all_t.T / TEMP
    labels = torch.arange(z.size(0), device=z.device)
    loss = F.cross_entropy(logits, labels)
    top1 = (logits.argmax(dim=1) == labels).float().mean().item()
    return loss, top1


def log(rec):
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("[stage0] " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                    for k, v in rec.items() if not isinstance(v, dict)}), flush=True)


# ------------------------------------------------------------------ data
xfit.patch_dataset_split(load_dataset, args.dataset)
ds = load_dataset[args.dataset]()
split = ds.get_idx_split()
train_loader = DataLoader([ds[i] for i in split["train"]], batch_size=args.batch_size,
                          shuffle=True, collate_fn=collate_fn, drop_last=True)
val_items = [ds[i] for i in split["val"]]
val_loader = DataLoader(val_items, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
print(f"[stage0] train {len(split['train'])} | val {len(split['val'])} | steps/epoch {len(train_loader)}")


def within_between(zn, gids):
    """Does the representation depend on the GRAPH or only on the question?

    SceneGraphs asks several questions about the same image (2.1 on average), so tokens can be
    split into pairs that share a graph and pairs that do not. If the representation is
    graph-driven, same-graph pairs are far more similar than different-graph pairs. If it is
    question-driven, the two are equal and a low inter-sample cosine means nothing about the
    graph. Undefined when every graph has one question (ExplaGraphs): reported as NaN.
    """
    import collections
    S = zn @ zn.T
    n = S.size(0)
    groups = collections.defaultdict(list)
    for i, g in enumerate(gids):
        groups[g].append(i)
    same = [S[i, j].item() for idx in groups.values() if len(idx) > 1
            for a, i in enumerate(idx) for j in idx[a + 1:]]
    if not same:
        return dict(within_graph_cos=float("nan"), between_graph_cos=float("nan"),
                    graph_gap=float("nan"), n_multi_q_graphs=0)
    tot = (S.sum() - n) / 2
    between = float((tot - sum(same)) / max((n * (n - 1)) / 2 - len(same), 1))
    within = float(sum(same) / len(same))
    return dict(within_graph_cos=within, between_graph_cos=between,
                graph_gap=within - between,
                n_multi_q_graphs=sum(1 for v in groups.values() if len(v) > 1))


@torch.no_grad()
def validate():
    """Full-split metrics: top-1 is over the ENTIRE validation set, not per batch."""
    encoder.eval()
    Z, T, GID = [], [], []
    for b in val_loader:
        GID += [str(x) for x in b.get("image_id", b["id"])]
        Z.append(encode(b).float().cpu())
        txt, keys = target_text(b)
        T.append(sbert(txt, keys=keys).float().cpu())
    encoder.train()
    Z, T = torch.cat(Z), torch.cat(T)
    zn, tn = F.normalize(Z, dim=-1), F.normalize(T, dim=-1)
    logits = zn @ tn.T
    labels = torch.arange(Z.size(0))
    out = dict(val_loss=F.cross_entropy(logits / TEMP, labels).item(),
               val_top1=(logits.argmax(1) == labels).float().mean().item(),
               val_token=stats(Z), val_target=stats(T))
    out.update(within_between(zn, GID))
    return out


# ------------------------------------------------------------------ train
bank = deque(maxlen=max(BANK // max(args.batch_size, 1), 1))
best, step, t0 = -1.0, 0, time.time()
v = validate()
log(dict(step=0, epoch=-1, phase="init", target=TARGET,
         **{k: v[k] for k in ("val_loss", "val_top1", "within_graph_cos", "between_graph_cos", "graph_gap")},
         val_cos=v["val_token"]["cos_inter"], val_cv=v["val_token"]["cv"],
         val_eff_rank=v["val_token"]["eff_rank"], target_cos=v["val_target"]["cos_inter"]))

for epoch in range(args.num_epochs):
    for batch in train_loader:
        z = encode(batch)
        txt, keys = target_text(batch)
        t = sbert(txt, keys=keys)
        bank_t = torch.cat(list(bank)) if bank else None
        loss, top1 = infonce(z, t, bank_t)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        bank.append(t.detach())
        if step % EVERY == 0:
            s = stats(z.detach())
            log(dict(step=step, epoch=epoch, phase="train", loss=loss.item(), top1=top1,
                     n_neg=int(z.size(0) + (bank_t.size(0) if bank_t is not None else 0)),
                     cos=s["cos_inter"], cv=s["cv"], eff_rank=s["eff_rank"],
                     seconds=round(time.time() - t0, 1)))
        step += 1
    v = validate()
    log(dict(step=step, epoch=epoch, phase="val",
             **{k: v[k] for k in ("val_loss", "val_top1", "within_graph_cos", "between_graph_cos", "graph_gap")},
             val_cos=v["val_token"]["cos_inter"], val_cv=v["val_token"]["cv"],
             val_resid=v["val_token"]["resid_over_mean"], val_eff_rank=v["val_token"]["eff_rank"],
             seconds=round(time.time() - t0, 1)))
    if v["val_top1"] > best:
        best = v["val_top1"]
        torch.save(dict(graph_encoder=encoder.state_dict(), W_q=W_q.state_dict(),
                        qcond=QCOND, epoch=epoch, val_top1=best,
                        gnn=dict(name=args.gnn_model_name, in_dim=args.gnn_in_dim,
                                 hidden_dim=args.gnn_hidden_dim, num_layers=args.gnn_num_layers,
                                 dropout=args.gnn_dropout, num_heads=args.gnn_num_heads)), CKPT)
        print(f"[stage0] saved {CKPT} (epoch {epoch}, val_top1 {best:.4f})", flush=True)
print(f"[stage0] done. best val_top1={best:.4f}  checkpoint={CKPT}  log={LOG}")
