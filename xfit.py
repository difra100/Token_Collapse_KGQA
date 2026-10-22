"""Cross-fitting split shared by every Paper-B stage.

Stage 1 memorises its training answers (train top-1 ~0.9 vs val ~0.5), so a Stage-2/3 model
trained on the SAME questions learns to copy the answer out of the graph token. Splitting the
training set fixes it: Stage 1 trains on half A, Stages 2-3 (and the probe's train-graph
statistics) on half B, so the graph tokens seen in training behave like test tokens.

ALIGN_TRAIN_HALF=A|B selects the half; unset = the full train split (all older runs).
The split is a fixed-seed permutation of the sorted train ids, so every script, every process
and every rerun gets identical, disjoint halves. Validation and test splits are never touched.
"""
import os
import random


def half(ids, which, seed=0):
    ids = sorted(int(i) for i in ids)
    perm = ids[:]
    random.Random(seed).shuffle(perm)
    h = len(perm) // 2
    if which == "A":
        return sorted(perm[:h])
    if which == "B":
        return sorted(perm[h:])
    raise ValueError(f"ALIGN_TRAIN_HALF must be A or B, got {which!r}")


def half_grouped(ids, group_of, which, seed=0):
    """As half(), but the split is over GROUPS: every id of a group lands in the same half.

    SceneGraphs asks several questions about one scene, so a question-level split would put the
    same graph in both halves and Stage 1 would still have seen half B's graphs.
    """
    ids = sorted(int(i) for i in ids)
    groups = sorted({group_of(i) for i in ids})
    perm = groups[:]
    random.Random(seed).shuffle(perm)
    h = len(perm) // 2
    if which not in ("A", "B"):
        raise ValueError(f"ALIGN_TRAIN_HALF must be A or B, got {which!r}")
    keep = set(perm[:h] if which == "A" else perm[h:])
    return [i for i in ids if group_of(i) in keep]


def half_for(name, ids, which, dataset=None):
    """The split every stage must use for this dataset: grouped by scene for SceneGraphs."""
    if name == "scene_graphs" and dataset is not None:
        col = dataset.questions["image_id"]
        return half_grouped(ids, lambda i: int(col.iloc[i]), which)
    return half(ids, which)


def which():
    w = os.environ.get("ALIGN_TRAIN_HALF", "").strip().upper()
    return w or None


def patch_dataset_split(load_dataset, name):
    """Make <dataset>.get_idx_split() return only the selected train half (val/test unchanged)."""
    w = which()
    if not w:
        return None
    cls = load_dataset[name]
    if getattr(cls, "_xfit_patched", None):
        return w
    orig = cls.get_idx_split

    def get_idx_split(self):
        s = dict(orig(self))
        full = len(s["train"])
        s["train"] = half_for(name, s["train"], w, self)
        print(f"[xfit] {name}: train split restricted to half {w} ({len(s['train'])} of {full})")
        return s
    cls.get_idx_split = get_idx_split
    cls._xfit_patched = w
    return w
