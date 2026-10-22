"""Graph-token probe for upstream G-Retriever checkpoints (Paper B, fixed weights).

Runs on a trained `graph_llm` checkpoint from paperB/G-Retriever-upstream WITHOUT
modifying that repo: the model is built and loaded with the repo's own code
(parse_args_llama, load_model, _reload_best_model, GraphLLM.inference, eval_funcs); the
interventions are applied from outside by wrapping two callables on the loaded instance:
  - model.encode_graphs : input-side corruption (node features replaced, structure kept)
  - model.projector     : token-side replacement (after the projector = the graph token)
and by editing the sample dict (text removed: desc = "").

Conditions (all on the test split; accuracy with the repo's eval_funcs):
  text present
    real                  unchanged model
    mean_token            every token := mean real token over the TRAIN split
    shuffle_token         token of another test sample (fixed derangement)
    zero_token            token := 0
    residual_x0 / x2      mu + a*(t_i - mu), a in {0, 2}  (x0 == mean_token, kept as check)
    rand_nodes_real       node features := rows drawn from the pool of train-graph node
                          features (realistic values, structure and node count kept)
    rand_nodes_gauss      node features := N(per-dim train mean, per-dim train std)
  text absent (desc = "")
    real, mean_token, shuffle_token, zero_token, rand_nodes_real
Token statistics (real test tokens): cos_inter, CV, eff_rank (mean-centred), norms,
||t_i - mu|| / ||mu||, and the LLM input-embedding (vocabulary) norm distribution;
input sensitivity cos(t, t_corrupt) and relative change for the rand_nodes conditions.

  python token_probe.py --gnn gt --dataset expla_graphs [--limit 32] [--out DIR]
"""
import argparse, json, os, sys, time

UP = os.environ.get("GRAPH_PROMPT_UPSTREAM", os.path.join(os.path.dirname(os.path.abspath(__file__)), "G-Retriever-upstream"))
ap = argparse.ArgumentParser()
ap.add_argument("--gnn", required=True, choices=["gt", "gcn", "gat", "mlp", "const", "constnorm", "align", "align_gelu", "align_relu", "align_none", "align_tanh", "align_gelu_pdrop50", "align_gelu_tdrop40"])
ap.add_argument("--model-name", default="graph_llm", choices=["graph_llm", "mlpq_llm", "const_llm", "constnorm_llm", "align_llm", "gelu_llm", "align3_llm", "alignjoint_llm", "aligndrop_llm", "alignjointtd_llm", "flex_graph_llm", "flex_gelu_graph_llm"])
ap.add_argument("--dataset", default="expla_graphs")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--limit", type=int, default=0, help="first N test samples (0 = all)")
ap.add_argument("--train-sample", type=int, default=2000,
                help="train graphs used for the mean token and the node-feature pool (0 = all)")
ap.add_argument("--conditions", default="all")
ap.add_argument("--llm", default="7b", help="the repo's --llm_model_name; extra: mistral7b")
ap.add_argument("--ckpt-dir", default="output", help="the repo's --output_dir used at training time")
ap.add_argument("--num-epochs", default="10", help="the --num_epochs used at training time (part of the checkpoint name)")
ap.add_argument("--patience", default=None,
                help="only needed when training used a NON-default patience. config.py declares "
                     "patience as type=float with default 2, and argparse does not convert defaults: "
                     "not passing it gives the int 2 -> 'patience_2' (what every older run wrote), "
                     "while passing it gives a float -> 'patience_4.0'. So leave this unset for legacy "
                     "checkpoints and pass e.g. 4.0 only for runs trained with --patience 4.")
ap.add_argument("--nll", action="store_true",
                help="measure the gold-answer negative log-likelihood per question instead of "
                     "decoding; detects effects on the output distribution that accuracy hides")
ap.add_argument("--llm-frozen", default="True", choices=["True", "False"],
                help="the repo's --llm_frozen used at training time (False = LoRA)")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
ORIG_ARGV = sys.argv[1:]
cli = ap.parse_args()
cli.out = os.path.abspath(cli.out)      # children chdir into the repo; keep one output location

# One fresh process per condition. Within a single process, repeated identical LLM calls
# drift (GPU kernel selection depends on allocator state; forcing deterministic algorithms
# does not remove it): the SAME condition run twice changed 1/64 answers for the trained GT
# and 4/16 for a boundary model, while separate processes are bit-identical. Every
# condition therefore runs as the first inference call after an identical load + stats pass.
COND_NAMES = ["real", "mean_token", "shuffle_token", "zero_token", "residual_x0", "residual_x2",
              "rand_nodes_real", "rand_nodes_gauss", "notext_real", "notext_mean_token",
              "notext_shuffle_token", "notext_zero_token", "notext_rand_nodes_real",
              "norm_vocab", "scale_x0.5", "scale_x2", "notext_norm_vocab",
              "rand_edges", "rand_edge_attr",
              # same question, another question's whole graph: isolates the graph from the
              # question conditioning, which SWAP cannot do
              "graph_swap", "notext_graph_swap",
              # two further SWAP permutations, to show the effect is not one lucky pairing
              "shuffle_token_p2", "shuffle_token_p3"]
LLM_PATHS = {"mistral7b": "mistralai/Mistral-7B-v0.1", "qwen3b": "Qwen/Qwen2.5-3B"}   # keep in sync with controls.LLM_PATHS
ODIR = os.path.join(cli.out, cli.dataset, ("" if cli.model_name == "graph_llm" else cli.model_name + "_")
                    + cli.gnn + (f"_n{cli.limit}" if cli.limit else "")
                    + (f"_seed{cli.seed}" if cli.seed != 0 else "")    # seed 0 keeps the existing path
                    + (f"_llm-{cli.llm}" if cli.llm != "7b" else "")
                    + ("_lora" if cli.llm_frozen == "False" else ""))
PARTS = os.path.join(ODIR, "parts")
if not cli.child:
    import subprocess
    os.makedirs(PARTS, exist_ok=True)
    wanted = COND_NAMES if cli.conditions == "all" else cli.conditions.split()
    base, skip = [], False
    for a in ORIG_ARGV:                      # drop --conditions X, re-added per child
        if skip:
            skip = False; continue
        if a == "--conditions":
            skip = True; continue
        if a.startswith("--conditions="):
            continue
        base.append(a)
    me = os.path.abspath(__file__)
    # the statistics job is complete only once the raw test tokens are saved too (jackknife CIs)
    pre = "nll_" if cli.nll else ""
    jobs = [(n, os.path.join(PARTS, f"{pre}{n}.json")) for n in wanted] + \
           [("none", os.path.join(PARTS, "_tokens.pt"))]
    for n, part in jobs:
        if os.path.exists(part):
            print(f"[isolate] {n}: done already"); continue
        print(f"[isolate] {n}: fresh process ...", flush=True)
        r = subprocess.run([sys.executable, me, *base, "--conditions", n, "--child"])
        if r.returncode != 0 or not os.path.exists(part):
            sys.exit(f"[isolate] condition {n} FAILED (exit {r.returncode}); rerun resumes from here")
    acc = {n: json.load(open(os.path.join(PARTS, f"{pre}{n}.json"))) for n in wanted}
    meta = json.load(open(os.path.join(PARTS, "_token_stats.json")))
    summary = dict(meta["meta"], accuracy=acc, token_stats=meta["token_stats"],
                   isolation="one fresh process per condition")
    with open(os.path.join(ODIR, "summary_nll.json" if cli.nll else "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if cli.nll:
        print(f"\n{'condition':26} gold-answer NLL")
        for n in wanted:
            print(f"{n:26} {acc[n]['nll_mean']:.4f}  (n={acc[n]['n']})")
        print(f"-> {ODIR}/summary_nll.json")
        sys.exit(0)
    print(f"\n{'condition':26} acc")
    for n in wanted:
        extra = (f"  yes/no {acc[n]['acc_yesno']:.4f}  open {acc[n]['acc_open']:.4f}" if "acc_open" in acc[n] else "")
        print(f"{n:26} {acc[n]['acc']:.4f}{extra}")
    print(f"-> {ODIR}/summary.json")
    sys.exit(0)

HERE = os.path.dirname(os.path.abspath(__file__))   # before chdir: __file__ may be relative
os.chdir(UP)                    # the repo's datasets use relative paths
sys.path.insert(0, UP)
sys.argv = [sys.argv[0], "--dataset", cli.dataset, "--model_name", cli.model_name,
            "--gnn_model_name", cli.gnn, "--seed", str(cli.seed), "--max_memory", "80",
            "--output_dir", cli.ckpt_dir, "--num_epochs", cli.num_epochs, "--llm_model_name", cli.llm,
            "--llm_frozen", cli.llm_frozen]
if cli.patience is not None:          # see --patience help: passing it changes int 2 -> float 2.0
    sys.argv += ["--patience", str(cli.patience)]

import pandas as pd                                           # noqa: E402
import torch                                                  # noqa: E402
from torch.utils.data import DataLoader                       # noqa: E402
from src.config import parse_args_llama                       # noqa: E402
from src.dataset import load_dataset                          # noqa: E402
from src.model import load_model, llama_model_path            # noqa: E402
from src.utils.ckpt import _reload_best_model                 # noqa: E402
from src.utils.collate import collate_fn                      # noqa: E402
from src.utils.evaluate import eval_funcs                     # noqa: E402
from src.utils.seed import seed_everything                    # noqa: E402
llama_model_path.update(LLM_PATHS)
if os.environ.get("GRAPH_PROMPT_LLM_PATH"):
    llama_model_path[cli.llm] = os.environ["GRAPH_PROMPT_LLM_PATH"]
sys.path.insert(0, HERE)
import controls                                               # noqa: E402
import public_models                                          # noqa: E402
controls.register()
public_models.register()     # graph-free controls + projector sized to the LLM (no-op for 4096-dim LLMs)

args = parse_args_llama()
seed_everything(seed=args.seed)
args.llm_model_path = llama_model_path[args.llm_model_name]
dataset = load_dataset[args.dataset]()
split = dataset.get_idx_split()
test_ids = split["test"][: cli.limit] if cli.limit else split["test"]
train_ids = split["train"]
import xfit                                   # mean token / node pools from the Stage-2 half
if xfit.which():
    train_ids = xfit.half_for(args.dataset, train_ids, xfit.which(), dataset)
    print(f"[xfit] probe train statistics from half {xfit.which()} ({len(train_ids)} graphs)")
if cli.train_sample and len(train_ids) > cli.train_sample:
    _g = torch.Generator().manual_seed(0)
    train_ids = [train_ids[i] for i in torch.randperm(len(train_ids), generator=_g)[: cli.train_sample].tolist()]
print(f"[data] {args.dataset}: {len(test_ids)} test samples, {len(train_ids)} train graphs for mean/pool")

model = load_model[args.model_name](graph_type=dataset.graph_type, args=args, init_prompt=dataset.prompt)

# ---- load the checkpoint with the repo's function, then make sure nothing was skipped
ckpt_path = (f"{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_name_"
             f"{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_"
             f"max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_"
             f"{args.patience}_num_epochs_{args.num_epochs}_seed{args.seed}_checkpoint_best.pth")
if not os.path.exists(ckpt_path):
    sys.exit(f"checkpoint not found: {ckpt_path}")
ck_keys = set(torch.load(ckpt_path, map_location="cpu")["model"].keys())
trainable = {n for n, p in model.named_parameters() if p.requires_grad}
model_keys = set(model.state_dict().keys())
if not ck_keys <= model_keys:
    sys.exit(f"checkpoint keys missing in model: {sorted(ck_keys - model_keys)[:5]}")
if not trainable <= ck_keys:
    sys.exit(f"trainable params absent from checkpoint: {sorted(trainable - ck_keys)[:5]}")
before = {n: p.detach().clone() for n, p in model.named_parameters() if n in trainable}
model = _reload_best_model(model, args)
changed = sum(not torch.equal(before[n], p.detach()) for n, p in model.named_parameters() if n in before)
print(f"[load] {len(ck_keys)} checkpoint tensors; {len(trainable)} trainable; {changed} changed by load")
if changed != len(trainable):
    sys.exit("some trainable parameters were not overwritten by the checkpoint -- aborting")
model.eval()
dev = model.model.device
bs = args.eval_batch_size


def loader(ids):
    return DataLoader([dataset[i] for i in ids], batch_size=bs, shuffle=False,
                      drop_last=False, pin_memory=True, collate_fn=collate_fn)


# ---- originals and intervention wrappers --------------------------------------------
orig_encode = model.encode_graphs
orig_projector = model.projector
STATE = {"node_mode": None, "token_mode": None, "batch_ids": None}


@torch.no_grad()
def pooled_and_tokens(ids, node_mode=None):
    """(g_embeds, tokens) for samples `ids` under an input corruption, no LLM call."""
    gs, ts = [], []
    for b in loader(ids):
        g = encode_with(b, node_mode)
        gs.append(g.float().cpu()); ts.append(orig_projector(g).float().cpu())
    return torch.cat(gs), torch.cat(ts)


# pools for input corruptions: train node features and edge attributes
_xs, _eas = [], []
for _i in train_ids:
    _g = dataset[_i]["graph"]
    _xs.append(_g.x.float())
    if getattr(_g, "edge_attr", None) is not None and _g.edge_attr.numel():
        _eas.append(_g.edge_attr.float())
train_x = torch.cat(_xs)
train_ea = torch.cat(_eas) if _eas else None
x_mu, x_sd = train_x.mean(0), train_x.std(0)
gen = torch.Generator().manual_seed(1234)


def corrupt_x(x, mode):
    n = x.size(0)
    if mode == "real_rows":
        idx = torch.randint(0, train_x.size(0), (n,), generator=gen)
        return train_x[idx].to(x.dtype)
    if mode == "gauss":
        return (x_mu + x_sd * torch.randn(n, x.size(1), generator=gen)).to(x.dtype)
    raise ValueError(mode)


def encode_with(samples, node_mode):
    if node_mode is None:
        return orig_encode(samples)
    g = samples["graph"].clone()
    if node_mode == "edges_random":
        # rewire every edge uniformly inside its own graph; edge count, node count, node and
        # edge features kept
        ei = g.edge_index.cpu()
        if ei.numel():
            gid = g.batch.cpu()[ei[0]]
            start = g.ptr.cpu()[gid]
            size = (g.ptr.cpu()[1:] - g.ptr.cpu()[:-1])[gid]
            new = start.unsqueeze(0) + (torch.rand(2, ei.size(1), generator=gen) * size.unsqueeze(0)).long()
            g.edge_index = new.to(g.edge_index.dtype)
    elif node_mode == "graph_swap":
        # replace the whole graph with another question's graph; the question conditioning inside
        # the encoder still uses THIS question, so any effect is attributable to the graph
        from torch_geometric.data import Batch
        partners = [dataset[test_ids[perm[pos[i]].item()]]["graph"] for i in samples["id"]]
        s2 = dict(samples); s2["graph"] = Batch.from_data_list(partners)
        return orig_encode(s2)
    elif node_mode == "edge_attr_real":
        if train_ea is not None and getattr(g, "edge_attr", None) is not None and g.edge_attr.numel():
            idx = torch.randint(0, train_ea.size(0), (g.edge_attr.size(0),), generator=gen)
            g.edge_attr = train_ea[idx].to(g.edge_attr.dtype)
    else:
        g.x = corrupt_x(g.x.cpu(), node_mode)
    s2 = dict(samples); s2["graph"] = g
    return orig_encode(s2)


print("[stats] computing train mean token and real test tokens ...")
_, train_tok = pooled_and_tokens(train_ids)
mu = train_tok.mean(0)
_, test_tok = pooled_and_tokens(test_ids)
pos = {sid: k for k, sid in enumerate(test_ids)}
perm = torch.roll(torch.arange(len(test_ids)), shifts=1)        # derangement: i -> i-1


def _derangement(n, seed):
    """A permutation with no fixed point, so every question really gets another one's token.

    Rejection sampling: a random permutation is a derangement with probability about 1/e, so a
    handful of draws suffices, and the result is uniform over derangements.
    """
    g = torch.Generator().manual_seed(seed)
    for _ in range(1000):
        p = torch.randperm(n, generator=g)
        if bool((p != torch.arange(n)).all()):
            return p
    raise RuntimeError(f"no derangement found for n={n}, seed={seed}")


perm2, perm3 = _derangement(len(test_ids), 7), _derangement(len(test_ids), 13)


class TokenIntervention(torch.nn.Module):
    def forward(self, g):
        t = orig_projector(g)
        mode, ids = STATE["token_mode"], STATE["batch_ids"]
        if mode is None:
            return t
        if mode == "mean":
            return mu.to(t.dtype).to(t.device).expand_as(t).clone()
        if mode == "zero":
            return torch.zeros_like(t)
        if mode in ("shuffle", "shuffle2", "shuffle3"):
            pm = {"shuffle": perm, "shuffle2": perm2, "shuffle3": perm3}[mode]
            k = torch.tensor([pm[pos[i]].item() for i in ids])
            return test_tok[k].to(t.dtype).to(t.device)
        if mode == "normvocab":        # direction kept, norm := median input-embedding norm
            return torch.nn.functional.normalize(t.float(), dim=-1).to(t.dtype) * VOCAB_MEDIAN
        if mode.startswith("scale"):
            return t * float(mode[len("scale"):])
        if mode.startswith("resid"):
            a = float(mode[len("resid"):])
            m = mu.to(t.dtype).to(t.device)
            return m + a * (t - m)
        raise ValueError(mode)


def patched_encode(samples):
    STATE["batch_ids"] = list(samples["id"])
    return encode_with(samples, STATE["node_mode"])


VOCAB_MEDIAN = model.word_embedding.weight.detach().float().norm(dim=-1).median().item()
model.encode_graphs = patched_encode
model.projector = TokenIntervention()

CONDS = {  # name: (text, node_mode, token_mode)
    "real": (True, None, None), "mean_token": (True, None, "mean"),
    "shuffle_token": (True, None, "shuffle"), "zero_token": (True, None, "zero"),
    "residual_x0": (True, None, "resid0"), "residual_x2": (True, None, "resid2"),
    "rand_nodes_real": (True, "real_rows", None), "rand_nodes_gauss": (True, "gauss", None),
    "notext_real": (False, None, None), "notext_mean_token": (False, None, "mean"),
    "notext_shuffle_token": (False, None, "shuffle"), "notext_zero_token": (False, None, "zero"),
    "notext_rand_nodes_real": (False, "real_rows", None),
    # H3: magnitude, direction kept
    "norm_vocab": (True, None, "normvocab"), "scale_x0.5": (True, None, "scale0.5"),
    "scale_x2": (True, None, "scale2"), "notext_norm_vocab": (False, None, "normvocab"),
    # which graph property the token depends on: structure / relation attributes
    "rand_edges": (True, "edges_random", None), "rand_edge_attr": (True, "edge_attr_real", None),
    "graph_swap": (True, "graph_swap", None), "notext_graph_swap": (False, "graph_swap", None),
    "shuffle_token_p2": (True, None, "shuffle2"), "shuffle_token_p3": (True, None, "shuffle3"),
}
assert list(CONDS) == COND_NAMES, "COND_NAMES out of sync with CONDS"
names = [] if cli.conditions == "none" else cli.conditions.split()
odir = ODIR
os.makedirs(odir, exist_ok=True)

results = {}
for name in names:
    text, node_mode, token_mode = CONDS[name]
    STATE.update(node_mode=node_mode, token_mode=token_mode)
    gen.manual_seed(1234)
    if cli.nll:
        # per-question gold-answer NLL under teacher forcing: a finer measure than accuracy,
        # because the token can move the output distribution without flipping correctness
        path = os.path.join(odir, f"nll_{name}.jsonl")
        t0 = time.time()
        vals = []
        with open(path, "w") as f, torch.no_grad():
            for b in DataLoader([dataset[i] for i in test_ids], batch_size=1, shuffle=False,
                                collate_fn=collate_fn):
                if not text:
                    b = dict(b); b["desc"] = [""] * len(b["id"])
                STATE["batch_ids"] = list(b["id"])
                v = float(model(b))
                vals.append(v)
                f.write(json.dumps({"id": int(b["id"][0]), "nll": v}) + "\n")
        results[name] = {"nll_mean": sum(vals) / len(vals), "n": len(vals),
                         "seconds": round(time.time() - t0, 1)}
        print(f"  {name:24} NLL {results[name]['nll_mean']:.4f}  ({results[name]['seconds']}s)",
              flush=True)
        with open(os.path.join(PARTS, f"nll_{name}.json"), "w") as f:
            json.dump(results[name], f)
        continue
    path = os.path.join(odir, f"{name}.jsonl")
    t0 = time.time()
    with open(path, "w") as f, torch.no_grad():
        for b in loader(test_ids):
            if not text:
                b = dict(b); b["desc"] = [""] * len(b["id"])
            out = model.inference(b)
            for _, row in pd.DataFrame(out).iterrows():
                f.write(json.dumps(dict(row)) + "\n")
    acc = eval_funcs[args.dataset](path)
    results[name] = {"acc": acc, "seconds": round(time.time() - t0, 1)}
    if args.dataset == "scene_graphs":
        # the repo's rule (label substring of pred), split into yes/no vs open questions:
        # yes/no has a 0.5 coin-flip floor and can hide what happens on open questions
        rows = [json.loads(l) for l in open(path)]
        yn = [r for r in rows if str(r["label"]).strip().lower() in ("yes", "no")]
        op = [r for r in rows if str(r["label"]).strip().lower() not in ("yes", "no")]
        hit = lambda rs: sum(str(r["label"]) in str(r["pred"]) for r in rs) / max(len(rs), 1)
        results[name].update(acc_yesno=hit(yn), acc_open=hit(op), n_yesno=len(yn), n_open=len(op))
    extra = (f"  yes/no {results[name]['acc_yesno']:.4f}  open {results[name]['acc_open']:.4f}"
             if "acc_open" in results[name] else "")
    print(f"  {name:24} acc {acc:.4f}{extra}  ({results[name]['seconds']}s)", flush=True)
    with open(os.path.join(PARTS, f"{name}.json"), "w") as f:
        json.dump(results[name], f)
STATE.update(node_mode=None, token_mode=None)
if names:
    sys.exit(0)            # an inference child does not recompute token statistics

# ---- token statistics -----------------------------------------------------------------
def stats(T, word_embedding_norm=None):
    Tn = torch.nn.functional.normalize(T, dim=-1)
    n = T.size(0)
    cos = (Tn @ Tn.T).sum().sub(n).div(n * (n - 1)).item()
    cv = (T.std(0).mean() / T.abs().mean()).item()
    s = torch.linalg.svdvals(T - T.mean(0))
    p = s / s.sum()
    er = torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()).item()
    residual_norm_mean = (T - T.mean(0)).norm(dim=-1).mean()
    out = dict(cos_inter=cos, cv=cv, eff_rank=er, norm_mean=T.norm(dim=-1).mean().item(),
               resid_over_mean=(residual_norm_mean / T.mean(0).norm()).item(),
               residual_norm_mean=residual_norm_mean.item())
    if word_embedding_norm is not None:
        out["residual_over_word_embedding"] = (residual_norm_mean / word_embedding_norm).item()
    return out


W = model.word_embedding.weight.detach().float().cpu()
vn = W.norm(dim=-1)
sim = torch.nn.functional.normalize(mu, dim=0) @ torch.nn.functional.normalize(W, dim=-1).T
top = sim.topk(5)
tok_stats = {
    "test_tokens": stats(test_tok, vn.median()),
    "vocab_norm": {"mean": vn.mean().item(), "median": vn.median().item(),
                   "p01": vn.quantile(0.01).item(), "p99": vn.quantile(0.99).item(), "max": vn.max().item()},
    "token_norm_over_vocab_median": (test_tok.norm(dim=-1).mean() / vn.median()).item(),
    "mean_token_nearest_vocab": [(model.tokenizer.convert_ids_to_tokens(int(i)), round(float(v), 4))
                                 for v, i in zip(top.values, top.indices)],
}
for mode in ("real_rows", "gauss", "edges_random", "edge_attr_real"):
    gen.manual_seed(1234)
    _, tc = pooled_and_tokens(test_ids, mode)
    tok_stats[f"sensitivity_{mode}"] = {
        "cos_mean": torch.nn.functional.cosine_similarity(test_tok, tc, dim=-1).mean().item(),
        "rel_change": ((tc - test_tok).norm(dim=-1) / test_tok.norm(dim=-1)).mean().item(),
        "corrupt_tokens": stats(tc, vn.median())}

meta = {"dataset": args.dataset, "model_name": cli.model_name, "gnn": cli.gnn, "seed": args.seed, "llm": cli.llm,
        "n_test": len(test_ids), "n_train_for_mean": len(train_ids), "checkpoint": ckpt_path}
with open(os.path.join(PARTS, "_token_stats.json"), "w") as f:
    json.dump({"meta": meta, "token_stats": tok_stats}, f, indent=2)
torch.save({"test_tokens": test_tok, "train_mean": mu, "test_ids": list(test_ids)}, os.path.join(PARTS, "_tokens.pt"))
print(json.dumps(tok_stats, indent=1))
