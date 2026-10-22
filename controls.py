"""Graph-free controls for upstream G-Retriever (Paper B), registered from outside the repo.

Each control subclasses the repo's own GraphLLM and changes ONLY where the pre-projector
vector comes from; LLM, prompt layout, projector (Linear 1024->2048, Sigmoid, Linear
2048->4096), losses, optimiser and training loop are the repo's, untouched:

  mlpq_llm   g = MLP(sbert(question))   the question string exactly as fed to the LLM,
             encoded with the repo's own sentence encoder (all-roberta-large-v1, 1024-d, the
             one used for node features); MLP = gnn_num_layers Linear layers of width
             gnn_hidden_dim with ReLU/dropout, i.e. the GCN's depth and width without edges.
             The graph is never read.
  const_llm  g = a learned 1024-d vector, identical for every sample. The projector is kept,
             so this matches G-Retriever's parametrisation of the token with zero input
             information (unlike pt_llm's raw soft prompt).
  constnorm_llm  as const_llm, but the projector output is rescaled to the median norm of
             the LLM's input embeddings at every forward (training and test): the same
             parametrisation with the magnitude removed.  (H3/H4)
  ptbig_llm  the repo's PromptTuningLLM (10 soft tokens, no projector), with the soft tokens
             initialised from the same prompt word embeddings but rescaled to norm
             PT_INIT_NORM = 21.6 (the trained constant token's norm). Everything else
             is pt_llm. Tests whether magnitude alone closes the pt_llm gap.  (H4)

Use with register(); checkpoint/CSV names get model_name mlpq_llm/const_llm and
gnn_model_name mlp/const, so they never collide with graph_llm runs.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import load_model
from src.model.graph_llm import GraphLLM
from src.model.pt_llm import PromptTuningLLM
from src.utils.lm_modeling import load_sbert, sber_text2embedding

_SBERT = {}          # kept outside the nn.Module: frozen, never saved in checkpoints


def _sbert():
    if "m" not in _SBERT:
        m, tok, dev = load_sbert()
        for p in m.parameters():
            p.requires_grad_(False)
        _SBERT.update(m=m, tok=tok, dev=dev, cache={})
    return _SBERT


class _MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers, dropout):
        super().__init__()
        dims = [in_dim] + [hidden] * (num_layers - 1) + [out_dim]
        self.lins = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.dropout = dropout

    def forward(self, x):
        for lin in self.lins[:-1]:
            x = F.dropout(F.relu(lin(x)), p=self.dropout, training=self.training)
        return self.lins[-1](x)


class _NoGraphLLM(GraphLLM):
    """Build the repo's GraphLLM (with a throwaway GT so its __init__ runs unchanged), then
    drop the graph encoder."""
    GNN_TAG = None

    def __init__(self, args, **kwargs):
        tag = args.gnn_model_name
        args.gnn_model_name = "gt"          # GraphLLM.__init__ looks the name up in load_gnn_model
        super().__init__(args=args, **kwargs)
        args.gnn_model_name = tag
        if tag != self.GNN_TAG:
            raise ValueError(f"{type(self).__name__} requires --gnn_model_name {self.GNN_TAG}, got {tag}")
        del self.graph_encoder


class QuestionMLPLLM(_NoGraphLLM):
    GNN_TAG = "mlp"

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.graph_encoder = _MLP(args.gnn_in_dim, args.gnn_hidden_dim, args.gnn_hidden_dim,
                                  args.gnn_num_layers, args.gnn_dropout).to(self.model.device)

    def encode_graphs(self, samples):
        s = _sbert()
        todo = [(i, q) for i, q in zip(samples["id"], samples["question"]) if i not in s["cache"]]
        if todo:
            emb = sber_text2embedding(s["m"], s["tok"], s["dev"], [q for _, q in todo])
            for (i, _), e in zip(todo, emb):
                s["cache"][i] = e
        q = torch.stack([s["cache"][i] for i in samples["id"]]).to(self.model.device).float()
        return self.graph_encoder(q)


class ConstTokenLLM(_NoGraphLLM):
    GNN_TAG = "const"

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.const = nn.Parameter(torch.randn(args.gnn_hidden_dim, device=self.model.device) * 0.02)

    def encode_graphs(self, samples):
        return self.const.unsqueeze(0).expand(len(samples["id"]), -1)


class _NormTo(nn.Module):
    """Wraps the repo's projector; output direction kept, norm set to `target`."""
    def __init__(self, projector, target):
        super().__init__()
        self.projector = projector
        self.target = float(target)

    def forward(self, x):
        return F.normalize(self.projector(x), dim=-1) * self.target


class ConstWordNormLLM(ConstTokenLLM):
    GNN_TAG = "constnorm"

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        target = self.word_embedding.weight.detach().float().norm(dim=-1).median().item()
        self.projector = _NormTo(self.projector, target)
        print(f"[constnorm] projector output held at norm {target:.4f} (median input-embedding norm)")


PT_INIT_NORM = 21.6


class BigInitPromptLLM(PromptTuningLLM):
    def __init__(self, init_prompt, args, **kwargs):
        super().__init__(init_prompt=init_prompt, args=args, **kwargs)
        with torch.no_grad():
            before = self.prompt.norm(dim=-1).mean().item()
            self.prompt.mul_(PT_INIT_NORM / self.prompt.norm(dim=-1, keepdim=True))
        print(f"[ptbig] soft prompt rescaled at init: mean norm {before:.4f} -> "
              f"{self.prompt.norm(dim=-1).mean().item():.4f}")



# ----------------------------------------------------------------- Paper B stage 2
class AlignGraphLLM(GraphLLM):
    """Stage 2: the encoder comes from stage-0 alignment (stage0_align.py) and is FROZEN;
    only G-Retriever's projector trains, with the repo's own QA loss and training loop.

    PAPER_SKELETON 5.5: a frozen pretrained encoder with a projector-only budget gave
    cos .846 / eff-rank 75.7 and significant shuffle and random-node effects, while the
    same encoder fine-tuned through the QA loss degraded to cos .939. This class is that
    setting, with the pretraining replaced by the self-supervised alignment objective.

    The checkpoint path comes from ALIGN_CKPT. Question conditioning is replayed exactly
    as in stage 0 (x + W_q(sbert(question))), so the encoder sees identical inputs in both
    stages; W_q is frozen too. train.py:49 builds the optimiser from requires_grad, and
    line 55 prints the trainable count -- check it is the projector's 12,593,152 alone.
    """
    GNN_TAG = "align"          # + "_<act>" when the projector activation is not the repo's

    def __init__(self, args, **kwargs):
        tag = args.gnn_model_name
        # The activation is part of the tag, so it lands in the checkpoint filename. A probe
        # run with a different ALIGN_PROJ then cannot find the weights instead of silently
        # loading them into the wrong architecture (the shapes are identical either way).
        want_act = os.environ.get("ALIGN_PROJ", "sigmoid").lower()
        expect = self.GNN_TAG if want_act == "sigmoid" else f"{self.GNN_TAG}_{want_act}"
        if tag != expect:
            raise ValueError(f"ALIGN_PROJ={want_act} requires --gnn_model_name {expect}, got {tag}")
        args.gnn_model_name = "gt"            # the architecture stage 0 pretrained
        super().__init__(args=args, **kwargs)
        args.gnn_model_name = tag

        path = os.environ.get("ALIGN_CKPT", "")
        # train_control.py chdir's into the upstream repo, so resolve a relative path
        # against this file's directory (paperB/) rather than the cwd.
        if path and not os.path.isabs(path):
            here = os.path.dirname(os.path.abspath(__file__))
            for cand in (os.path.join(here, path),
                         os.path.join(here, os.path.basename(os.path.dirname(path)), os.path.basename(path)),
                         os.path.join(os.path.dirname(here), path)):
                if os.path.exists(cand):
                    path = cand
                    break
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"set ALIGN_CKPT to a stage0_align.py checkpoint (got {path!r})")
        ck = torch.load(path, map_location="cpu")
        for k, want in (("in_dim", args.gnn_in_dim), ("hidden_dim", args.gnn_hidden_dim),
                        ("num_layers", args.gnn_num_layers), ("num_heads", args.gnn_num_heads)):
            if ck["gnn"][k] != want:
                raise ValueError(f"stage-0 checkpoint gnn.{k}={ck['gnn'][k]} != this run's {want}")
        self.graph_encoder.load_state_dict(ck["graph_encoder"])   # strict: a mismatch must raise
        self.qcond = bool(ck["qcond"])
        self.W_q = nn.Linear(args.gnn_in_dim, args.gnn_in_dim).to(self.model.device)
        self.W_q.load_state_dict(ck["W_q"])
        for p in list(self.graph_encoder.parameters()) + list(self.W_q.parameters()):
            p.requires_grad_(False)
        act = want_act
        if act != "sigmoid":
            gnn_hidden = self.projector[0].in_features
            out = self.projector[-1].out_features
            mid = {"sigmoid": nn.Sigmoid, "gelu": nn.GELU, "relu": nn.ReLU,
                   "tanh": nn.Tanh, "none": nn.Identity}[act]
            self.projector = nn.Sequential(nn.Linear(gnn_hidden, 2048), mid(),
                                           nn.Linear(2048, out)).to(self.model.device)
            print(f"[align] projector non-linearity Sigmoid -> {act} (same shapes, same param count)")
        print(f"[align] encoder loaded from {path} (epoch {ck['epoch']}, val_top1 {ck['val_top1']:.4f}), "
              f"frozen; qcond={self.qcond}; only the projector trains")

    def encode_graphs(self, samples):
        from torch_scatter import scatter
        graphs = samples["graph"].to(self.model.device)
        # ALIGN_NOGRAPH=1 zeroes the node features, so the encoder sees only the question
        # through W_q: the control that asks whether the remedy needs the graph at all.
        x = torch.zeros_like(graphs.x) if os.environ.get("ALIGN_NOGRAPH", "0") == "1" else graphs.x
        if self.qcond:
            s = _sbert()
            todo = [(i, q) for i, q in zip(samples["id"], samples["question"]) if ("q", i) not in s["cache"]]
            if todo:
                emb = sber_text2embedding(s["m"], s["tok"], s["dev"], [q for _, q in todo])
                for (i, _), e in zip(todo, emb):
                    s["cache"][("q", i)] = e
            q = torch.stack([s["cache"][("q", i)] for i in samples["id"]]).to(self.model.device).float()
            x = x + self.W_q(q)[graphs.batch]
        n_embeds, _ = self.graph_encoder(x, graphs.edge_index.long(), graphs.edge_attr)
        return scatter(n_embeds, graphs.batch, dim=0, reduce="mean")



class GeluProjGraphLLM(GraphLLM):
    """Upstream G-Retriever, unchanged, EXCEPT the projector's non-linearity.

    G-Retriever's projector is Linear(gnn_hidden, 2048) -> Sigmoid -> Linear(2048, 4096).
    With moderate inputs the Sigmoid sits near 0.5 for every graph, so the token is close
    to the last layer's bias whatever the graph is. This class swaps that one activation
    (PROJ_ACT, default gelu) and changes nothing else: same shapes, same parameter count,
    same encoder (trainable, from scratch), same losses, same training loop.

    It exists to answer whether the Sigmoid alone explains collapse in the ORIGINAL
    system -- the align_llm evidence only shows it for a pretrained frozen encoder.
    Registered as model_name 'gelu_llm', so its checkpoints and results never collide
    with the graph_llm runs already on disk.
    """

    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        act = os.environ.get("PROJ_ACT", "gelu").lower()
        mid = {"sigmoid": nn.Sigmoid, "gelu": nn.GELU, "relu": nn.ReLU,
               "tanh": nn.Tanh, "none": nn.Identity}[act]
        gnn_hidden, out = self.projector[0].in_features, self.projector[-1].out_features
        self.projector = nn.Sequential(nn.Linear(gnn_hidden, 2048), mid(),
                                       nn.Linear(2048, out)).to(self.model.device)
        print(f"[gelu_llm] projector Sigmoid -> {act}; everything else is upstream G-Retriever")


class PromptPlusGraphLLM(AlignGraphLLM):
    """Stage 3: a trainable prompt token next to the frozen, non-collapsed graph token.

    Motivation (PAPER_B_PLAN §8-9): a collapsed token earns G-Retriever's accuracy as a learned
    prompt -- a constant through the projector matches it (85.6 ExplaGraphs, 80.8 SceneGraphs) --
    while making the token graph-dependent costs accuracy because one token has to do both jobs
    and the QA loss drags it back towards a constant. Here the jobs are split:

      frozen   : Stage-1 encoder + W_q (AlignGraphLLM) and the Stage-2 projector (STAGE2_CKPT)
                 -> the graph token, which therefore cannot drift or re-collapse
      trainable: one learned 1024-d vector -> its own Linear -> Sigmoid -> Linear
                 -> the prompt token. Exactly the const_llm parametrisation, NOT a raw soft
                 token: raw prompt tuning is 10-25 points worse than the constant.

    LLM input: [BOS, prompt token, graph token, text...]. forward/inference are upstream's
    (src/model/graph_llm.py:118-216) with that one insertion changed. The graph token is still
    self.projector(self.encode_graphs(samples)), so token_probe's interventions (which patch
    model.projector and model.encode_graphs) act on the graph token only, never on the prompt.
    """

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        path = os.environ.get("STAGE2_CKPT", "")
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"set STAGE2_CKPT to the Stage-2 checkpoint_best.pth (got {path!r})")
        sd = torch.load(path, map_location="cpu")["model"]
        proj = {k[len("projector."):]: v for k, v in sd.items() if k.startswith("projector.")}
        want = set(self.projector.state_dict())
        if set(proj) != want:
            raise ValueError(f"Stage-2 projector keys {sorted(proj)} != model projector keys {sorted(want)}")
        self.projector.load_state_dict(proj)            # strict
        for p in self.projector.parameters():
            p.requires_grad_(False)
        # The graph transformer has BatchNorm layers. requires_grad=False freezes their weights but
        # NOT their running statistics, which Stage 2 kept updating and saved in its checkpoint.
        # Load those, so the graph token is exactly Stage 2's, and keep the encoder in eval mode
        # (see train() below) so they cannot drift again during Stage 3.
        enc = {k[len("graph_encoder."):]: v for k, v in sd.items() if k.startswith("graph_encoder.")}
        buffers = dict(self.graph_encoder.named_buffers())
        if set(enc) - set(buffers):
            raise ValueError(f"unexpected graph_encoder tensors in the Stage-2 checkpoint: {sorted(set(enc) - set(buffers))[:5]}")
        with torch.no_grad():
            for k, v in enc.items():
                buffers[k].copy_(v.to(buffers[k].device))
        self.graph_encoder.eval()
        out = self.projector[-1].out_features
        self.prompt_const = nn.Parameter(torch.randn(args.gnn_hidden_dim, device=self.model.device) * 0.02)
        self.prompt_projector = nn.Sequential(nn.Linear(args.gnn_hidden_dim, 2048), nn.Sigmoid(),
                                              nn.Linear(2048, out)).to(self.model.device)
        print(f"[stage3] Stage-2 projector loaded from {path} and frozen ({len(proj)} tensors); "
              f"trainable prompt token = constant -> Linear -> Sigmoid -> Linear; "
              f"encoder BatchNorm statistics from Stage 2 ({len(enc)} buffers), encoder locked in eval mode")

    def train(self, mode=True):
        super().train(mode)
        self.graph_encoder.eval()        # frozen means frozen: no BatchNorm statistic updates
        return self

    def _prefix(self, samples):
        g = self.projector(self.encode_graphs(samples))               # (B, H) graph tokens
        p = self.prompt_projector(self.prompt_const).unsqueeze(0).expand_as(g)
        return torch.stack([p, g], dim=1)                             # (B, 2, H)

    def forward(self, samples):
        from src.model.graph_llm import EOS, EOS_USER, BOS, IGNORE_INDEX
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
        labels = self.tokenizer(samples["label"], add_special_tokens=False)
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)
        prefix = self._prefix(samples)
        batch_size = len(samples['id'])
        batch_inputs_embeds, batch_attention_mask, batch_label_input_ids = [], [], []
        for i in range(batch_size):
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids
            input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, prefix[i], inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0] - len(label_input_ids)) + label_input_ids
            batch_label_input_ids.append(label_input_ids)
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length + batch_label_input_ids[i]
        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)
        with self.maybe_autocast():
            outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                 return_dict=True, labels=label_input_ids)
        return outputs.loss

    def inference(self, samples):
        from src.model.graph_llm import EOS_USER, BOS
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)
        prefix = self._prefix(samples)
        batch_size = len(samples['id'])
        batch_inputs_embeds, batch_attention_mask = [], []
        for i in range(batch_size):
            input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, prefix[i], inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        with self.maybe_autocast():
            outputs = self.model.generate(inputs_embeds=inputs_embeds, max_new_tokens=self.max_new_tokens,
                                          attention_mask=attention_mask, use_cache=True)
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return {'id': samples['id'], 'pred': pred, 'label': samples['label'],
                'question': samples['question'], 'desc': samples['desc']}


class AlignJointLLM(AlignGraphLLM):
    """Stage 2 with the Stage-3 prompt token trained JOINTLY from step 0.

    Why: in the cross-fitted WebQSP run Stage 3 started from an already-fitted Stage 2 (train loss
    .79) and its best epoch was 0, so the prompt token never took over the steering role. Here the
    prompt token (constant -> Linear -> Sigmoid -> Linear, as in PromptPlusGraphLLM) and the graph
    projector train together from the start. Encoder + W_q frozen exactly as in AlignGraphLLM
    (BatchNorm statistics adapt, as in Stage 2). Prefix, forward and inference are
    PromptPlusGraphLLM's: [BOS, prompt token, graph token, text].
    """

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        out = self.projector[-1].out_features
        self.prompt_const = nn.Parameter(torch.randn(args.gnn_hidden_dim, device=self.model.device) * 0.02)
        self.prompt_projector = nn.Sequential(nn.Linear(args.gnn_hidden_dim, 2048), nn.Sigmoid(),
                                              nn.Linear(2048, out)).to(self.model.device)
        print("[joint] prompt token (constant -> Linear -> Sigmoid -> Linear) and graph projector "
              "trained jointly from step 0; encoder frozen as in Stage 2")

    _prefix = PromptPlusGraphLLM._prefix
    forward = PromptPlusGraphLLM.forward
    inference = PromptPlusGraphLLM.inference


class TextDropJointLLM(AlignJointLLM):
    """AlignJointLLM whose TRAINING prompts keep only a fraction of the verbalised edges.

    Why: every model we measured ignores the graph's structure, although the token carries it
    (rewiring the edges changes our token by >100% and accuracy by nothing). With the whole graph
    in the text there is no pressure to use the token. Here a fraction TEXT_DROP of the edge lines
    is removed from `desc` during training only, so the token is the only source for those facts.
    Evaluation is untouched (inference() is not overridden), i.e. the test prompt is the full graph.

    Only the edge block is touched ("\\n\\nsrc,edge_attr,dst\\n" + one line per edge, as written by
    every G-Retriever dataset); the node block and the question are left as they are, and at least
    one edge always survives. Uses a private RNG, so the training loop's random streams are
    untouched. p is part of the gnn tag (align_gelu_tdrop40) so checkpoints never collide.
    """
    EDGE_HEADER = "\n\nsrc,edge_attr,dst\n"

    def __init__(self, args, **kwargs):
        p = float(os.environ["TEXT_DROP"])
        if not 0.0 < p < 1.0:
            raise ValueError(f"TEXT_DROP must be in (0,1), got {p}")
        tag = args.gnn_model_name
        base = tag.rsplit("_tdrop", 1)[0]
        if tag != f"{base}_tdrop{round(p * 100)}":
            raise ValueError(f"TEXT_DROP={p} requires --gnn_model_name {base}_tdrop{round(p * 100)}, got {tag}")
        args.gnn_model_name = base
        super().__init__(args, **kwargs)
        args.gnn_model_name = tag
        self.text_drop = p
        import random as _random
        self._rng = _random.Random(1234 + int(args.seed))
        print(f"[tdrop] training prompts keep {(1-p)*100:.0f}% of the verbalised edges; "
              f"evaluation uses the full graph")

    def _drop_edges(self, desc):
        head, sep, edges = desc.partition(self.EDGE_HEADER)
        if not sep:
            raise ValueError("desc has no edge block; TextDropJointLLM cannot be used with this dataset")
        lines = [l for l in edges.split("\n") if l]
        keep = [l for l in lines if self._rng.random() >= self.text_drop]
        if lines and not keep:
            keep = [self._rng.choice(lines)]
        return head + sep + "\n".join(keep) + ("\n" if keep else "")

    def forward(self, samples):
        samples = dict(samples)
        samples["desc"] = [self._drop_edges(d) for d in samples["desc"]]
        return super().forward(samples)


class ProjDropGraphLLM(AlignGraphLLM):
    """Stage 2 with dropout on the projector's hidden layer (PROJ_DROP, required).

    Why: at matched train loss the cross-fitted Stage 2 has val loss 1.23 vs 1.04 for G-Retriever,
    i.e. the non-collapsed token lets the (frozen) LLM fit training questions. With the LLM frozen
    the projector is the only trainable part, so regularisation has to act there. (Replacing the
    token by the mean during training would not: dropped rows send no gradient to the projector.)
    Projector = Linear -> act -> Dropout(p) -> Linear; dropout is off in eval (inference, probe,
    dynamics). p is part of the gnn tag (align_gelu_pdrop50) so checkpoints never collide.
    """

    def __init__(self, args, **kwargs):
        p = float(os.environ["PROJ_DROP"])
        tag = args.gnn_model_name
        base = tag.rsplit("_pdrop", 1)[0]
        if tag != f"{base}_pdrop{round(p * 100)}":
            raise ValueError(f"PROJ_DROP={p} requires --gnn_model_name {base}_pdrop{round(p * 100)}, got {tag}")
        args.gnn_model_name = base
        super().__init__(args, **kwargs)
        args.gnn_model_name = tag
        if len(self.projector) != 3:
            raise ValueError(f"expected a 3-module projector, got {self.projector}")
        lin1, act, lin2 = self.projector
        self.projector = nn.Sequential(lin1, act, nn.Dropout(p), lin2)
        print(f"[pdrop] projector = Linear -> {type(act).__name__} -> Dropout({p}) -> Linear")


LLM_PATHS = {"mistral7b": "mistralai/Mistral-7B-v0.1",
             "qwen3b": "Qwen/Qwen2.5-3B"}          # hidden 2048: projector output resized below


def _match_projector_to_llm():
    """G-Retriever hard-codes the projector output at 4096 (Llama-2-7b's hidden size). For an LLM
    with another hidden size, rebuild the SAME architecture with the last layer sized to the LLM:
    Linear(gnn_hidden, 2048) -> Sigmoid -> Linear(2048, hidden). Applied to GraphLLM and therefore
    to every control that subclasses it; a no-op for 4096-dim LLMs (Llama-2, Mistral)."""
    if getattr(GraphLLM, "_projector_patched", False):
        return
    _orig = GraphLLM.__init__

    def _init(self, *a, **k):
        _orig(self, *a, **k)
        hidden = self.word_embedding.weight.shape[1]
        if hidden != self.projector[-1].out_features:
            gnn_hidden = self.projector[0].in_features
            self.projector = nn.Sequential(nn.Linear(gnn_hidden, 2048), nn.Sigmoid(),
                                           nn.Linear(2048, hidden)).to(self.model.device)
            print(f"[controls] projector output resized 4096 -> {hidden} to match the LLM")
    GraphLLM.__init__ = _init
    GraphLLM._projector_patched = True


def register():
    _match_projector_to_llm()
    load_model["mlpq_llm"] = QuestionMLPLLM
    load_model["const_llm"] = ConstTokenLLM
    load_model["constnorm_llm"] = ConstWordNormLLM
    load_model["ptbig_llm"] = BigInitPromptLLM
    load_model["align_llm"] = AlignGraphLLM
    load_model["gelu_llm"] = GeluProjGraphLLM
    load_model["align3_llm"] = PromptPlusGraphLLM
    load_model["alignjoint_llm"] = AlignJointLLM
    load_model["aligndrop_llm"] = ProjDropGraphLLM
    load_model["alignjointtd_llm"] = TextDropJointLLM
