"""Manim animation for "On the Token Collapse Phenomenon in Textual Graph Soft Prompting".

Render one scene (low quality preview):
    .venv/bin/manim -ql token_collapse.py Pipeline
Render the full video in 1080p60:
    .venv/bin/manim -qh token_collapse.py FullVideo

No LaTeX is needed: every label uses Text (Pango), never MathTex.
Point clouds are schematic (synthetic), scaled to the statistics reported in the paper;
every number shown on screen is taken from the paper.
"""

from manim import *
from manim.animation.animation import prepare_animation
import numpy as np

# ---------------------------------------------------------------- style
BG = "#0E1116"
FG = "#E8E6E3"
MUTED = "#8B949E"
FAINT = "#30363D"
ORANGE = "#F28C28"   # trained modules (as in the paper's Figure 2)
BLUE = "#4C9BE8"     # frozen modules
RED = "#E5534B"      # collapse
GREEN = "#3FB950"    # our pipeline
YELLOW = "#E3B341"
PALETTE = ["#F28C28", "#4C9BE8", "#3FB950", "#C678DD", "#E5534B", "#E3B341", "#56B6C2"]
FONT = "DejaVu Sans"

config.background_color = BG


def T(s, size=26, color=FG, weight=NORMAL, **kw):
    return Text(s, font=FONT, font_size=size, color=color, weight=weight, **kw)


def box(label, color, w=2.4, h=0.9, size=22, fill=0.18):
    r = RoundedRectangle(corner_radius=0.15, width=w, height=h,
                         stroke_color=color, stroke_width=3,
                         fill_color=color, fill_opacity=fill)
    t = T(label, size=size)
    if t.width > w - 0.2:
        t.scale_to_fit_width(w - 0.2)
    return VGroup(r, t.move_to(r))


def chip(label, color, size=18, h=0.5, pad=0.22, fill=0.25):
    t = T(label, size=size)
    r = RoundedRectangle(corner_radius=0.08, width=t.width + 2 * pad, height=h,
                         stroke_color=color, stroke_width=2,
                         fill_color=color, fill_opacity=fill)
    return VGroup(r, t.move_to(r))


def section_title(scene, number, text):
    title = T(f"{number}  {text}".strip(), size=34, weight=BOLD).to_edge(UP, buff=0.4)
    line = Line(LEFT, RIGHT, color=FAINT).set_width(title.width + 0.6).next_to(title, DOWN, buff=0.15)
    scene.play(FadeIn(title, shift=0.2 * DOWN), Create(line), run_time=0.8)
    return VGroup(title, line)


def caption(text, size=24, color=FG):
    c = T(text, size=size, color=color, line_spacing=0.9)
    if c.width > 13:
        c.scale_to_fit_width(13)
    return c.to_edge(DOWN, buff=0.45)


def clear(scene, run_time=0.7):
    if scene.mobjects:
        scene.play(*[FadeOut(m) for m in scene.mobjects], run_time=run_time)


# Global pacing: every animation and pause lasts SLOW times longer.
SLOW = 1.25


class SlowScene(Scene):
    def play(self, *args, run_time=None, **kwargs):
        if run_time is None:
            args = [prepare_animation(a) for a in args]
            run_time = max(a.get_run_time() for a in args)
        super().play(*args, run_time=run_time * SLOW, **kwargs)

    def wait(self, duration=1.0, *args, **kwargs):
        super().wait(duration * SLOW, *args, **kwargs)


# ---------------------------------------------------------------- "soft prompt" highlight
SP_COLORS = ["#FFD86B", "#FF9F43", "#FF5E9C"]   # gold -> orange -> pink


def soft_prompt_badge(text="SOFT PROMPT", size=30, pad=0.28, height=None):
    """Bold gradient lettering on a glowing pill, used wherever the paper says 'soft prompt'."""
    word = Text(text, font=FONT, font_size=size, weight=BOLD)
    word.set_color_by_gradient(*SP_COLORS)
    h = height or word.height + 2 * pad * 0.8
    pill = RoundedRectangle(corner_radius=h / 2, width=word.width + 2 * pad, height=h,
                            stroke_width=2.5, fill_color="#2A1A10", fill_opacity=0.95)
    pill.set_stroke(color=SP_COLORS)
    glow = VGroup(*[pill.copy().set_fill(opacity=0).set_stroke(color=SP_COLORS[1], width=w, opacity=o)
                    for w, o in [(14, 0.08), (9, 0.12), (5, 0.2)]])
    return VGroup(glow, pill, word.move_to(pill))


def highlight_soft_prompt(s, badge, run_time=1.4):
    """Pop the badge in, sweep a light along its border, then pulse once."""
    glow, pill, word = badge
    s.play(FadeIn(glow), DrawBorderThenFill(pill), Write(word), run_time=run_time)
    flash = pill.copy().set_fill(opacity=0).set_stroke(color=WHITE, width=5)
    s.play(ShowPassingFlash(flash, time_width=0.35),
           badge.animate(rate_func=there_and_back).scale(1.08), run_time=0.9)


def mini_graph(seed, scale=0.35, color=BLUE):
    """A small random graph used as an 'input example'."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(4, 7))
    ang = np.sort(rng.uniform(0, TAU, n))
    rad = rng.uniform(0.6, 1.0, n)
    pts = [scale * np.array([r * np.cos(a), r * np.sin(a), 0]) for r, a in zip(rad, ang)]
    edges = [(i, (i + 1) % n) for i in range(n)]
    edges += [(int(rng.integers(n)), int(rng.integers(n))) for _ in range(2)]
    lines = VGroup(*[Line(pts[a], pts[b], stroke_width=2, color=MUTED)
                     for a, b in edges if a != b])
    dots = VGroup(*[Dot(p, radius=0.06, color=color) for p in pts])
    return VGroup(lines, dots)


def textual_graph():
    """The running example: a tiny WebQSP-style textual graph."""
    pos = {
        "Eiffel Tower": np.array([-1.3, 0.75, 0]),
        "Paris": np.array([0.15, -0.05, 0]),
        "France": np.array([1.35, 0.85, 0]),
        "Seine": np.array([-0.9, -0.95, 0]),
    }
    label_dir = {"Eiffel Tower": UP, "Paris": RIGHT, "France": UP, "Seine": DOWN}
    nodes, labels = VGroup(), VGroup()
    for name, p in pos.items():
        nodes.add(Dot(p, radius=0.11, color=BLUE))
        labels.add(T(name, size=16, color=FG).next_to(p, label_dir[name], buff=0.14))
    edge_spec = [("Eiffel Tower", "Paris", "located_in"),
                 ("Paris", "France", "capital_of"),
                 ("Seine", "Paris", "flows_through")]
    edges, elabels = VGroup(), VGroup()
    for a, b, rel in edge_spec:
        ln = Line(pos[a], pos[b], color=MUTED, stroke_width=2.5, buff=0.12)
        edges.add(ln)
        lab = T(rel, size=12, color=MUTED).move_to(ln.get_center())
        lab.shift(0.18 * rotate_vector(ln.get_unit_vector(), PI / 2))
        elabels.add(lab)
    return VGroup(edges, elabels, nodes, labels)


# ============================================================ act 0: title
def act_title(s: Scene):
    t1 = T("On the Token Collapse Phenomenon", size=46, weight=BOLD)
    t2 = T("in Textual Graph Soft Prompting", size=46, weight=BOLD)
    title = VGroup(t1, t2).arrange(DOWN, buff=0.2).shift(UP * 0.9)
    authors = T("A. G. Di Francesco · R. Jain · F. M. Nardini · N. Tonellotto · P. Liò · F. Silvestri",
                size=20, color=MUTED).next_to(title, DOWN, buff=0.6)
    affil = T("Sapienza · ISTI-CNR · Cambridge · Pisa", size=18, color=MUTED).next_to(authors, DOWN, buff=0.2)

    # many tokens converging into one: a preview of the story
    rng = np.random.default_rng(0)
    dots = VGroup(*[Dot(radius=0.05, color=PALETTE[i % len(PALETTE)]).move_to(
        np.array([rng.uniform(-6.5, 6.5), rng.uniform(-3.6, -1.9), 0])) for i in range(60)])
    s.play(FadeIn(title, shift=0.3 * UP), run_time=1.2)
    s.play(FadeIn(authors), FadeIn(affil), LaggedStart(*[FadeIn(d, scale=0.5) for d in dots],
                                                      lag_ratio=0.02), run_time=1.5)
    target = np.array([0, -2.75, 0])
    s.play(*[d.animate.move_to(target + 0.04 * rng.normal(size=3) * np.array([1, 1, 0])).set_color(RED)
             for d in dots], run_time=2, rate_func=rate_functions.ease_in_out_cubic)
    s.wait(0.8)
    clear(s)


# ============================================================ act 1: pipeline
def act_pipeline(s: Scene):
    head = section_title(s, "", "Graph soft prompting in G-Retriever")

    g = textual_graph().scale(0.9).move_to(LEFT * 5.0 + UP * 1.2)
    g_lab = T("textual graph Gᵢ", size=18, color=MUTED).next_to(g, DOWN, buff=0.25)
    s.play(FadeIn(g, lag_ratio=0.1), FadeIn(g_lab), run_time=1.4)

    enc = box("Graph encoder  fφ", ORANGE, w=2.5).move_to(LEFT * 1.6 + UP * 1.2)
    proj = box("Projector  Pψ", ORANGE, w=2.3).move_to(RIGHT * 2.1 + UP * 1.2)
    proj_sub = T("Linear → Sigmoid → Linear", size=14, color=MUTED).next_to(proj, DOWN, buff=0.12)
    token = Square(0.55, color=ORANGE, fill_opacity=0.85, stroke_width=0).move_to(RIGHT * 5.0 + UP * 1.2)
    token_lab = T("graph token tᵢ", size=18).next_to(token, DOWN, buff=0.15)

    a1 = Arrow(g.get_right() + RIGHT * 0.1, enc.get_left(), buff=0.1, color=MUTED, stroke_width=3)
    h_vec = VGroup(*[Square(0.18, stroke_width=1, color=FG, fill_color=c, fill_opacity=0.9)
                     for c in [BLUE, GREEN, ORANGE, YELLOW]]).arrange(DOWN, buff=0)
    h_vec.move_to((enc.get_right() + proj.get_left()) / 2)
    h_lab = T("hᵢ", size=18).next_to(h_vec, UP, buff=0.1)
    pool_lab = T("mean pool", size=13, color=MUTED).next_to(h_vec, DOWN, buff=0.1)
    a2 = Arrow(enc.get_right(), h_vec.get_left(), buff=0.08, color=MUTED, stroke_width=3)
    a3 = Arrow(h_vec.get_right(), proj.get_left(), buff=0.08, color=MUTED, stroke_width=3)
    a4 = Arrow(proj.get_right(), token.get_left(), buff=0.1, color=MUTED, stroke_width=3)

    s.play(GrowArrow(a1), FadeIn(enc), run_time=0.8)
    s.play(GrowArrow(a2), FadeIn(h_vec), FadeIn(h_lab), FadeIn(pool_lab), run_time=0.8)
    s.play(GrowArrow(a3), FadeIn(proj), FadeIn(proj_sub), run_time=0.8)
    s.play(GrowArrow(a4), FadeIn(token, scale=0.5), FadeIn(token_lab), run_time=0.8)

    # LLM input sequence
    bos = chip("<s>", MUTED, fill=0.15)
    t_chip = Square(0.5, color=ORANGE, fill_opacity=0.85, stroke_width=0)
    verb = VGroup(*[chip(x, BLUE, size=15) for x in
                    ["Eiffel Tower, located_in, Paris", "Paris, capital_of, France", "…"]])
    q = chip("Q: Which country is the Eiffel Tower in?", MUTED, size=15, fill=0.12)
    seq = VGroup(bos, t_chip, *verb, q).arrange(RIGHT, buff=0.12).scale_to_fit_width(12.2)
    seq.move_to(DOWN * 1.15 + LEFT * 0.4)
    brace_v = Brace(VGroup(*verb), DOWN, color=BLUE, buff=0.08)
    brace_v_t = T("verbalised graph τ(Gᵢ)", size=16, color=BLUE).next_to(brace_v, DOWN, buff=0.08)
    t_lab2 = T("tᵢ", size=16, color=ORANGE).next_to(t_chip, DOWN, buff=0.12)

    s.play(FadeIn(bos), run_time=0.3)
    s.play(TransformFromCopy(token, t_chip), FadeIn(t_lab2), run_time=1.0)
    s.play(LaggedStart(*[TransformFromCopy(g, v) for v in verb], lag_ratio=0.3),
           GrowFromCenter(brace_v), FadeIn(brace_v_t), run_time=1.4)
    s.play(FadeIn(q, shift=LEFT * 0.2), run_time=0.6)

    llm = box("Frozen LLM", BLUE, w=2.4, h=0.8).move_to(DOWN * 2.75 + RIGHT * 2.2)
    ans = T("France", size=26, color=GREEN, weight=BOLD).next_to(llm, RIGHT, buff=1.0)
    a5 = Arrow(np.array([llm.get_x(), seq.get_bottom()[1] - 0.05, 0]), llm.get_top(), buff=0.1,
               color=MUTED, stroke_width=3)
    a6 = Arrow(llm.get_right(), ans.get_left(), buff=0.1, color=MUTED, stroke_width=3)
    s.play(FadeIn(llm), GrowArrow(a5), run_time=0.7)
    s.play(GrowArrow(a6), Write(ans), run_time=0.7)

    cap = T("The same graph enters twice:\nas a learned token and as text.", size=24,
            color=YELLOW, line_spacing=1.0).move_to(DOWN * 2.85 + LEFT * 3.6)
    s.play(FadeIn(cap), Indicate(t_chip, color=ORANGE), Indicate(VGroup(*verb), color=BLUE),
           run_time=1.4)
    s.wait(1.5)
    claim = VGroup(
        T("G-Retriever interprets its ablations as evidence that", size=22, color=MUTED),
        T("the graph encoder and verbalised graph provide", size=28, weight=BOLD),
        T("complementary graph representations", size=28, weight=BOLD, color=ORANGE),
    ).arrange(DOWN, buff=0.16)
    claim_bg = Rectangle(width=config.frame_width, height=config.frame_height, stroke_width=0,
                         fill_color=BG, fill_opacity=0.88)
    s.play(FadeIn(claim_bg), FadeIn(claim, shift=0.15 * UP), run_time=1.0)
    s.wait(2.6)
    q2 = T("Does the graph token actually carry the graph?", size=28, weight=BOLD)
    q2_bg = BackgroundRectangle(q2, color=BG, fill_opacity=0.95, buff=0.35)
    s.play(FadeOut(claim), FadeIn(q2_bg), Write(q2), run_time=1.2)
    s.wait(1.5)
    clear(s)


# ============================================================ act 2: collapse
def act_collapse(s: Scene, number="1 ·"):
    head = section_title(s, number, "Graph tokens collapse")

    ax = Axes(x_range=[-4, 4, 1], y_range=[-3, 3, 1], x_length=7.2, y_length=5.0,
              axis_config={"color": FAINT, "include_ticks": False, "tip_length": 0.15})
    ax.move_to(LEFT * 2.6 + DOWN * 0.35)
    origin = ax.c2p(0, 0)
    s.play(Create(ax), run_time=0.8)

    # input graphs that become points
    graphs = VGroup(*[mini_graph(i, scale=0.32, color=PALETTE[i % 7]) for i in range(6)])
    graphs.arrange(RIGHT, buff=0.35).next_to(head, DOWN, buff=0.3).shift(LEFT * 2.6)
    in_lab = T("different input graphs", size=16, color=MUTED).next_to(graphs, RIGHT, buff=0.3)
    s.play(LaggedStart(*[FadeIn(gr, scale=0.6) for gr in graphs], lag_ratio=0.12), FadeIn(in_lab),
           run_time=1.2)

    # encoder outputs: spread out (encoder cosine ~0.59)
    rng = np.random.default_rng(3)
    N = 140
    ang = rng.uniform(-0.15, 1.75, N)
    rad = rng.uniform(1.0, 2.7, N)
    enc_xy = np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1) + rng.normal(0, 0.25, (N, 2))
    cols = [PALETTE[i % 7] for i in range(N)]
    dots = VGroup(*[Dot(ax.c2p(*enc_xy[i]), radius=0.05, color=cols[i], fill_opacity=0.9)
                    for i in range(N)])

    s.play(*[ReplacementTransform(graphs[i], dots[i]) for i in range(6)], FadeOut(in_lab), run_time=1.2)
    s.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots[6:]], lag_ratio=0.01), run_time=1.2)
    stage_lab = T("encoder outputs  hᵢ", size=22, color=FG).next_to(ax, UP, buff=0.05).align_to(ax, LEFT)
    s.play(FadeIn(stage_lab), run_time=0.5)

    # cosine readout panel
    panel_x = RIGHT * 4.3
    cos_title = T("mean pairwise cosine", size=20, color=MUTED).move_to(panel_x + UP * 1.7)
    cos_val = T("0.593", size=60, weight=BOLD, color=BLUE).next_to(cos_title, DOWN, buff=0.3)
    cos_note = T("encoder output (WebQSP, GT)", size=16, color=MUTED).next_to(cos_val, DOWN, buff=0.2)
    s.play(FadeIn(cos_title), FadeIn(cos_val), FadeIn(cos_note), run_time=0.7)
    s.wait(0.6)

    # projection: everything goes to mu + tiny residual
    mu = np.array([2.6, 1.9])
    # schematic residuals, scaled so their mean 2-D radius is 0.03 * |mu|, as in the paper's Figure 1
    res = rng.normal(0, 1, (N, 2))
    res *= 0.03 * np.linalg.norm(mu) / np.linalg.norm(res, axis=1).mean()
    tok_xy = mu + res
    proj_box = chip("projector  Pψ", ORANGE, size=18).next_to(ax, UP, buff=0.05).align_to(ax, RIGHT)
    s.play(FadeIn(proj_box, shift=0.2 * DOWN), run_time=0.5)
    new_lab = T("graph tokens  tᵢ", size=22, color=FG).move_to(stage_lab, aligned_edge=LEFT)
    new_val = T("0.996", size=60, weight=BOLD, color=RED).move_to(cos_val)
    new_note = T("graph token (WebQSP, GT)", size=16, color=MUTED).move_to(cos_note)
    s.play(*[dots[i].animate.move_to(ax.c2p(*tok_xy[i])) for i in range(N)],
           Transform(stage_lab, new_lab), Transform(cos_val, new_val), Transform(cos_note, new_note),
           run_time=2.4, rate_func=rate_functions.ease_in_out_cubic)
    mu_arrow = Arrow(origin, ax.c2p(*mu), buff=0.12, color=RED, stroke_width=4,
                     max_tip_length_to_length_ratio=0.08)
    mu_lab = T("shared mean μ", size=18, color=RED).next_to(mu_arrow.get_center(), DR, buff=0.08)
    s.play(GrowArrow(mu_arrow), FadeIn(mu_lab), run_time=0.9)
    s.wait(0.6)

    # zoom on the residual
    zoom_src = Square(0.45, color=YELLOW, stroke_width=2).move_to(ax.c2p(*mu))
    inset = Square(2.6, color=YELLOW, stroke_width=2, fill_color=BG, fill_opacity=1)
    inset.move_to(ax.c2p(-1.8, -1.2))
    # the inset is a true x6 magnification of the plotted points, clipped to the frame
    zres = np.clip(res * ax.x_length / 8 * 6, -1.2, 1.2)
    zoom_dots = VGroup(*[Dot(inset.get_center() + np.array([*zres[i], 0]),
                             radius=0.04, color=cols[i]) for i in range(N)])
    zl = T("zoom × 6", size=18, color=YELLOW, weight=BOLD).next_to(inset, UP, buff=0.08).align_to(inset, LEFT)
    links = VGroup(DashedLine(zoom_src.get_corner(DL), inset.get_corner(UR), color=YELLOW, stroke_width=1.5),
                   DashedLine(zoom_src.get_corner(DR), inset.get_corner(DR), color=YELLOW, stroke_width=1.5))
    s.play(Create(zoom_src), run_time=0.5)
    s.play(Create(links), FadeIn(inset), run_time=0.7)
    s.play(FadeIn(zoom_dots, lag_ratio=0.01), FadeIn(zl), run_time=1.0)
    rr = T("residual ratio  mean ‖tᵢ − μ‖ / ‖μ‖ = 0.057", size=15, color=YELLOW).next_to(inset, DOWN, buff=0.1)
    s.play(FadeIn(rr), run_time=0.6)
    schem = T("schematic points", size=13, color=MUTED).next_to(ax, DOWN, buff=0.05).align_to(ax, RIGHT)
    s.add(schem)

    gen = VGroup(
        T("Holds across", size=18, color=MUTED),
        T("3 benchmarks · 3 encoders", size=19),
        T("2 LLM families · frozen & LoRA", size=19),
        T("token cosine 0.985 – 0.998", size=19, color=RED),
        T("residual only 4–12% of μ", size=19, color=YELLOW),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to(panel_x + DOWN * 1.4)
    s.play(LaggedStart(*[FadeIn(x, shift=0.15 * RIGHT) for x in gen], lag_ratio=0.25), run_time=1.5)
    cap = caption("Every input gets almost the same token, and task performance does not reveal it.",
                  size=22, color=YELLOW)
    s.play(FadeIn(cap), run_time=0.7)
    s.wait(2.2)
    clear(s)


# ============================================================ act 3: interventions
def act_interventions(s: Scene):
    head = section_title(s, "2 ·", "What does the collapsed token do?")

    # mini sequence with a swappable token slot
    bos = chip("<s>", MUTED, fill=0.15, size=16)
    slot = Square(0.5, color=ORANGE, fill_opacity=0.85, stroke_width=0)
    verb = chip("verbalised graph τ(Gᵢ)", BLUE, size=16)
    q = chip("question qᵢ", MUTED, size=16, fill=0.12)
    seq = VGroup(bos, slot, verb, q).arrange(RIGHT, buff=0.12).move_to(UP * 1.75 + LEFT * 2.2)
    llm = box("LLM", BLUE, w=1.3, h=0.6, size=18).next_to(seq, RIGHT, buff=0.6)
    arr = Arrow(seq.get_right(), llm.get_left(), buff=0.08, color=MUTED, stroke_width=3)
    slot_lab = T("REAL  tᵢ", size=18, color=ORANGE).next_to(slot, UP, buff=0.15)
    s.play(FadeIn(seq), FadeIn(llm), GrowArrow(arr), FadeIn(slot_lab), run_time=0.9)

    # bar chart (WebQSP Hit@1, Llama-2, GT)
    base_y = -2.75
    scale_h = 2.8 / 80.0
    data = [("REAL", 70.33, ORANGE, "tᵢ"), ("MEAN", 68.49, MUTED, "μ"), ("SWAP", 69.41, MUTED, "tⱼ"),
            ("GSWAP", 69.78, MUTED, "t(G′)"), ("CONST", 71.25, GREEN, "c"), ("NOTXT", 41.34, RED, "")]
    xs = np.linspace(-5.6, 0.6, len(data))
    x0, x1 = -6.3, 1.3
    base = Line([x0, base_y, 0], [x1, base_y, 0], color=FAINT)
    ylab = T("WebQSP Hit@1  (Llama-2, GT)", size=16, color=MUTED).move_to([x0, base_y + 3.3, 0],
                                                                       aligned_edge=LEFT)
    s.play(Create(base), FadeIn(ylab), run_time=0.5)

    ref = DashedLine([x0, base_y + 70.33 * scale_h, 0], [x1, base_y + 70.33 * scale_h, 0],
                     color=ORANGE, stroke_width=1.5, dash_length=0.08)

    def bar(i):
        name, v, c, _ = data[i]
        r = Rectangle(width=0.8, height=v * scale_h, stroke_width=0, fill_color=c, fill_opacity=0.85)
        r.move_to([xs[i], base_y, 0], aligned_edge=DOWN)
        nl = T(name, size=15, color=FG).next_to(r, DOWN, buff=0.12)
        vl = T(f"{v:.2f}", size=15, color=FG).next_to(r, UP, buff=0.08)
        return VGroup(r, nl, vl)

    bars = [bar(i) for i in range(len(data))]
    descs = {
        "REAL": "the token computed from this graph",
        "MEAN": "replace it with the training-set mean token",
        "SWAP": "use another example's token",
        "GSWAP": "recompute it from a different graph",
        "CONST": "learned constant, never sees a graph",
        "NOTXT": "keep the token, delete the verbalised graph",
    }
    desc = T(descs["REAL"], size=20, color=MUTED).next_to(seq, DOWN, buff=0.35).align_to(seq, LEFT)
    s.play(GrowFromEdge(bars[0][0], DOWN), FadeIn(bars[0][1:]), FadeIn(desc), run_time=0.8)
    s.play(Create(ref), run_time=0.4)

    for i in range(1, 5):
        name, v, c, sym = data[i]
        new_slot = Square(0.5, color=c, fill_opacity=0.85, stroke_width=0).move_to(slot)
        sym_t = T(sym, size=16, color=BG, weight=BOLD).move_to(new_slot)
        new_lab = T(f"{name}  {sym}", size=18, color=c).move_to(slot_lab)
        new_desc = T(descs[name], size=20, color=MUTED).move_to(desc, aligned_edge=LEFT)
        s.play(Transform(slot, new_slot), FadeIn(sym_t), Transform(slot_lab, new_lab),
               Transform(desc, new_desc), run_time=0.6)
        s.play(GrowFromEdge(bars[i][0], DOWN), FadeIn(bars[i][1:]), run_time=0.6)
        s.remove(sym_t)
        s.wait(0.25)

    # NOTXT: remove the text instead
    new_desc = T(descs["NOTXT"], size=20, color=MUTED).move_to(desc, aligned_edge=LEFT)
    orig_slot = Square(0.5, color=ORANGE, fill_opacity=0.85, stroke_width=0).move_to(slot)
    s.play(Transform(slot, orig_slot), Transform(slot_lab, T("NOTXT", size=18, color=RED).move_to(slot_lab)),
           verb.animate.set_opacity(0.15), Transform(desc, new_desc), run_time=0.7)
    s.play(GrowFromEdge(bars[5][0], DOWN), FadeIn(bars[5][1:]), run_time=0.7)

    # takeaways on the right
    tk = VGroup(
        T("Replacing the token (MEAN/SWAP/GSWAP):", size=16, color=MUTED),
        T("within 2 points, no resolved effect", size=20, color=FG),
        T("Removing the verbalised graph:", size=17, color=MUTED),
        T("− 29 points", size=20, color=RED),
        T("Graph-free constant (CONST):", size=17, color=MUTED),
        T("matches or exceeds G-Retriever", size=20, color=GREEN),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.13)
    tk.move_to([1.7, -1.2, 0], aligned_edge=LEFT)
    for i in range(0, 6, 2):
        s.play(FadeIn(tk[i], shift=0.1 * RIGHT), FadeIn(tk[i + 1], shift=0.1 * RIGHT), run_time=0.7)
        s.wait(0.5)

    # verdict: dim the chart, then reveal the highlighted "soft prompt" statement
    s.wait(0.8)
    dim = Rectangle(width=config.frame_width, height=config.frame_height, stroke_width=0,
                    fill_color=BG, fill_opacity=0.82)
    l1 = T("The collapsed token works primarily as a", size=30, weight=BOLD)
    badge = soft_prompt_badge("SOFT PROMPT", size=44)
    l3 = T("not as a graph representation used to answer.", size=24, color=MUTED)
    verdict = VGroup(l1, badge, l3).arrange(DOWN, buff=0.35).move_to(DOWN * 0.2)
    s.play(FadeIn(dim), run_time=0.6)
    s.play(Write(l1), run_time=1.0)
    highlight_soft_prompt(s, badge)
    s.play(FadeIn(l3, shift=0.1 * UP), run_time=0.7)
    s.wait(2.2)
    clear(s)


# ============================================================ act 4: causes
def act_causes(s: Scene):
    head = section_title(s, "3 ·", "What is collapse associated with?")

    # left: the paper's Appendix H reading of the Sigmoid projector, sigma(x) ~ 1/2 + x/4 near zero
    ax = Axes(x_range=[-4, 4, 1], y_range=[-0.2, 1.2, 0.5], x_length=4.2, y_length=2.6,
              axis_config={"color": FAINT, "include_ticks": False, "tip_length": 0.12})
    ax.move_to(LEFT * 4.4 + UP * 0.75)
    curve = ax.plot(lambda x: 1 / (1 + np.exp(-x)), color=ORANGE, stroke_width=4)
    clab = T("Sigmoid in G-Retriever's projector", size=17, color=ORANGE).next_to(ax, UP, buff=0.1)
    s.play(Create(ax), Create(curve), FadeIn(clab), run_time=1.2)
    tangent = ax.plot(lambda x: 0.5 + x / 4, x_range=[-2.0, 2.0], color=YELLOW, stroke_width=2.5)
    tangent = DashedVMobject(tangent, num_dashes=24)
    tlab = T("σ(x) ≈ ½ + ¼x  near 0", size=15, color=YELLOW).next_to(ax.c2p(-4, 1.15), RIGHT, buff=0.05)
    s.play(Create(tangent), FadeIn(tlab), run_time=0.9)

    rng = np.random.default_rng(7)
    xs = rng.normal(0, 0.8, 18)
    in_dots = VGroup(*[Dot(ax.c2p(x, 0), radius=0.06, color=PALETTE[i % 7]) for i, x in enumerate(xs)])
    s.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in in_dots], lag_ratio=0.04), run_time=0.8)
    out_x = 4.4
    out_dots = VGroup(*[Dot(ax.c2p(out_x, 1 / (1 + np.exp(-x))), radius=0.06, color=PALETTE[i % 7])
                        for i, x in enumerate(xs)])
    s.play(*[TransformFromCopy(in_dots[i], out_dots[i], path_arc=0.6) for i in range(len(xs))], run_time=1.4)
    off = Brace(VGroup(Dot(ax.c2p(out_x, 0)), Dot(ax.c2p(out_x, 0.5))), RIGHT, color=RED, buff=0.15)
    offl = T("½: same for\nevery input", size=13, color=RED).next_to(off, RIGHT, buff=0.08)
    s.play(GrowFromCenter(off), FadeIn(offl), run_time=0.8)
    expl = T("The ½ term is input-independent and ¼ attenuates\n"
             "input-specific variation: one possible explanation (App. H).", size=14, color=MUTED)
    note = T("Frozen Stage-1 encoder outputs (cos 0.024),\nprojector before training (Table 3):",
             size=14, color=MUTED)
    nums = VGroup(T("Sigmoid → token cos 0.987", size=18, color=RED),
                  T("GELU → token cos 0.091", size=18, color=GREEN)).arrange(DOWN, buff=0.1)
    VGroup(expl, note, nums).arrange(DOWN, buff=0.18).next_to(ax, DOWN, buff=0.3).set_x(-3.6)
    s.play(FadeIn(expl), run_time=0.7)
    s.play(FadeIn(note), FadeIn(nums), run_time=0.7)
    s.wait(0.6)

    # right: the 2x2 table (ExplaGraphs, final token cosine)
    gx, gy = 4.4, 0.25
    cw, ch = 2.1, 1.25
    col_h = VGroup(T("Sigmoid", size=19, color=ORANGE), T("GELU", size=19, color=GREEN))
    row_h = VGroup(T("encoder trained\nend to end", size=16), T("encoder frozen\nafter Stage 1", size=16))
    vals = [[("0.997", "85.56", RED), ("0.914", "73.29", RED)],
            [("0.992", "84.48", RED), ("0.323", "57.76", GREEN)]]
    cells = VGroup()
    for r in range(2):
        for c in range(2):
            cen = np.array([gx + (c - 0.5) * cw, gy - (r - 0.5) * ch, 0])
            rect = Rectangle(width=cw, height=ch, stroke_color=FAINT, stroke_width=2).move_to(cen)
            v, sc, colr = vals[r][c]
            vt = T(v, size=30, weight=BOLD, color=colr).move_to(cen + UP * 0.15)
            st = T(f"acc {sc}", size=14, color=MUTED).move_to(cen + DOWN * 0.35)
            cells.add(VGroup(rect, vt, st))
    for c in range(2):
        col_h[c].move_to([gx + (c - 0.5) * cw, gy + ch + 0.05, 0])
    for r in range(2):
        row_h[r].move_to([gx - cw - 1.05, gy - (r - 0.5) * ch, 0])
    ttl = T("token cosine on ExplaGraphs", size=18, color=MUTED).next_to(col_h, UP, buff=0.2)
    s.play(FadeIn(ttl), FadeIn(col_h), FadeIn(row_h), *[Create(cl[0]) for cl in cells], run_time=0.9)
    for i in range(4):
        s.play(FadeIn(cells[i][1], scale=0.7), FadeIn(cells[i][2]), run_time=0.45)
    s.play(Indicate(cells[3], color=GREEN, scale_factor=1.06), run_time=0.9)
    warn = T("…but this single-token model falls to\n≈ majority-class accuracy (57.76 vs 58.8)", size=14,
             color=YELLOW)
    warn.next_to(cells, DOWN, buff=0.2).align_to(cells, RIGHT)
    s.play(FadeIn(warn), run_time=0.6)

    cap = caption("Two associated factors: a projector that suppresses encoder variation, and an answer\n"
                  "objective that does not require preserving it. Neither is isolated as the sole cause.",
                  size=20, color=YELLOW)
    s.play(FadeIn(cap), run_time=0.8)
    s.wait(2.5)
    clear(s)


# ============================================================ act 5: two-stage fix
def act_two_stage(s: Scene):
    head = section_title(s, "4 ·", "A two-stage pipeline that mitigates collapse")

    # ---- Stage 1: align the encoder without the LLM
    st1 = T("Stage 1  ·  align the graph encoder (no LLM)", size=21, weight=BOLD, color=ORANGE)
    st1.move_to([-6.6, 2.35, 0], aligned_edge=LEFT)
    y1 = 1.15
    g = mini_graph(11, scale=0.5, color=BLUE).move_to([-5.7, y1, 0])
    qv = T("question-conditioned:\nxᵥ = zᵥ + Wq qᵢ", size=13, color=MUTED).next_to(g, DOWN, buff=0.2)
    enc = box("Graph encoder fφ", ORANGE, w=2.4, h=0.8, size=18).move_to([-2.9, y1, 0])
    a1 = Arrow(g.get_right(), enc.get_left(), buff=0.12, color=MUTED, stroke_width=3)
    s.play(FadeIn(st1), FadeIn(g), FadeIn(qv), FadeIn(enc), GrowArrow(a1), run_time=1.0)

    # InfoNCE: pull h_i towards its target embedding, push the negatives away
    cen = np.array([1.6, y1, 0])
    space = RoundedRectangle(corner_radius=0.15, width=4.2, height=2.3, stroke_color=FAINT).move_to(cen)
    loss = T("InfoNCE", size=16, color=FG).next_to(space, UP, buff=0.08).align_to(space, RIGHT)
    h = Dot(cen + np.array([-1.2, -0.5, 0]), radius=0.09, color=ORANGE)
    hl = always_redraw(lambda: T("hᵢ", size=16, color=ORANGE).next_to(h, DOWN, buff=0.08))
    tgt = Star(5, outer_radius=0.15, color=GREEN, fill_opacity=1).move_to(cen + np.array([0.6, 0.35, 0]))
    tl = T("frozen target embedding", size=13, color=GREEN).next_to(tgt, UP, buff=0.08)
    negs = VGroup(*[Star(5, outer_radius=0.1, color=MUTED, fill_opacity=0.8).move_to(cen + np.array([dx, dy, 0]))
                    for dx, dy in [(-0.5, 0.55), (1.3, -0.55), (-1.5, 0.45), (0.3, -0.7)]])
    nl = T("negatives", size=13, color=MUTED).next_to(negs[3], RIGHT, buff=0.1)
    a2 = Arrow(enc.get_right(), space.get_left(), buff=0.1, color=MUTED, stroke_width=3)
    s.play(GrowArrow(a2), Create(space), FadeIn(loss), FadeIn(h), FadeIn(hl), FadeIn(tgt), FadeIn(tl),
           FadeIn(negs), FadeIn(nl), run_time=1.0)
    s.play(h.animate.move_to(tgt.get_center() + np.array([-0.3, -0.05, 0])),
           *[n.animate.shift(0.2 * normalize(n.get_center() - tgt.get_center())) for n in negs],
           run_time=1.3)

    frozen = box("Graph encoder fφ", BLUE, w=2.4, h=0.8, size=18).move_to(enc)
    fl = T("frozen", size=15, color=BLUE, weight=BOLD).next_to(enc, UP, buff=0.1)
    s.play(Transform(enc, frozen), FadeIn(fl), run_time=0.8)
    s.wait(0.4)

    # ---- Stage 2: two separate tokens trained with the answer loss
    st2 = T("Stage 2  ·  two separate tokens, original answer loss", size=21, weight=BOLD, color=ORANGE)
    st2.move_to([-6.6, -0.6, 0], aligned_edge=LEFT)
    s.play(FadeOut(VGroup(space, loss, h, hl, tgt, tl, negs, nl, a2)), FadeIn(st2), run_time=0.7)

    y2, y3 = -1.6, -2.75
    gelu = box("GELU projector", ORANGE, w=2.4, h=0.7, size=16).move_to([-2.9, y2, 0])
    a3 = Arrow(enc.get_bottom(), gelu.get_top(), buff=0.08, color=MUTED, stroke_width=3)
    pc = box("learned constant → Sigmoid proj.", ORANGE, w=3.6, h=0.7, size=15).move_to([-5.0, y3, 0])

    bos = chip("<s>", MUTED, fill=0.15, size=15, h=0.5)
    p_tok = Square(0.5, color=ORANGE, fill_opacity=0.9, stroke_width=0)
    g_tok = Square(0.5, color=GREEN, fill_opacity=0.9, stroke_width=0)
    verb = chip("τ(Gᵢ)", BLUE, size=15, h=0.5)
    qq = chip("qᵢ", MUTED, size=15, h=0.5, fill=0.12)
    seq = VGroup(bos, p_tok, g_tok, verb, qq).arrange(RIGHT, buff=0.1).move_to([1.6, (y2 + y3) / 2, 0])
    p_lab = T("p", size=15, color=ORANGE).next_to(p_tok, UP, buff=0.1)
    g_lab = T("tᵢ", size=15, color=GREEN).next_to(g_tok, DOWN, buff=0.1)
    a4 = CurvedArrow(gelu.get_right() + RIGHT * 0.08, g_tok.get_top() + UP * 0.05, angle=-PI / 3,
                     color=MUTED, stroke_width=3, tip_length=0.18)
    a5 = CurvedArrow(pc.get_right() + RIGHT * 0.08, p_tok.get_bottom() + DOWN * 0.05, angle=PI / 3,
                     color=MUTED, stroke_width=3, tip_length=0.18)
    llm = box("Frozen LLM", BLUE, w=1.9, h=0.7, size=16).next_to(seq, RIGHT, buff=0.6)
    a6 = Arrow(seq.get_right(), llm.get_left(), buff=0.08, color=MUTED, stroke_width=3)

    s.play(GrowArrow(a3), FadeIn(gelu), run_time=0.6)
    s.play(FadeIn(seq), Create(a4), FadeIn(g_lab), run_time=0.8)
    s.play(FadeIn(pc), Create(a5), FadeIn(p_lab), run_time=0.8)
    s.play(GrowArrow(a6), FadeIn(llm), run_time=0.6)
    roles = VGroup(T("tᵢ: keeps the input-specific variation", size=15, color=GREEN),
                   T("p: takes the shared prompting function", size=15, color=ORANGE)
                   ).arrange(DOWN, aligned_edge=LEFT, buff=0.08).next_to(llm, DOWN, buff=0.35).align_to(llm, RIGHT)
    s.play(FadeIn(roles), run_time=0.6)
    s.wait(2.0)
    s.play(*[FadeOut(m) for m in s.mobjects if not any(m is x for x in head)], run_time=0.7)

    # ---- Result: token clouds, same scale, under the same section title
    rng = np.random.default_rng(5)

    def cloud(center, spread, n=120):
        pts = np.clip(rng.normal(0, 1, (n, 2)) * spread, -1.4, 1.4)
        return VGroup(*[Dot(center + np.array([*p, 0]), radius=0.045, color=PALETTE[i % 7])
                        for i, p in enumerate(pts)])

    frames, labels = VGroup(), VGroup()
    centers = [np.array([-2.3, 0.2, 0]), np.array([2.3, 0.2, 0])]
    for c, (name, val, col) in zip(centers, [("G-Retriever", "cos 0.996", RED), ("Ours", "cos 0.142", GREEN)]):
        frames.add(RoundedRectangle(corner_radius=0.15, width=3.8, height=3.4, stroke_color=FAINT).move_to(c))
        labels.add(VGroup(T(name, size=20, color=col, weight=BOLD), T(val, size=18, color=col)
                          ).arrange(DOWN, buff=0.06).next_to(frames[-1], DOWN, buff=0.15))
    sub = T("WebQSP graph tokens minus their mean  (schematic points)", size=16, color=MUTED)
    sub.next_to(frames, UP, buff=0.15)
    unit = 0.85  # scene units per |mu|
    c_gr = cloud(centers[0], unit * 0.03 / np.sqrt(np.pi / 2))
    c_ou = cloud(centers[1], unit * 0.71 / np.sqrt(np.pi / 2))
    s.play(FadeIn(sub), Create(frames), run_time=0.8)
    s.play(FadeIn(c_gr), FadeIn(labels[0]), run_time=0.8)
    start = VGroup(*[d.copy().move_to(centers[1]) for d in c_ou])
    s.play(FadeIn(start), run_time=0.2)
    s.play(Transform(start, c_ou), FadeIn(labels[1]), run_time=1.5, rate_func=rate_functions.ease_out_cubic)

    cap = caption("Collapse is mitigated in every tested setting; performance stays competitive in most cases.", size=21,
                  color=YELLOW)
    s.play(FadeIn(cap), run_time=0.7)
    s.wait(2.5)
    clear(s)


# ============================================================ act 6: takeaway
def act_takeaway(s: Scene):
    head = section_title(s, "5 ·", "Distinct tokens are not necessarily used")

    qs = ["Does the token vary\nacross inputs?", "What does it\ncontain?", "Does the LLM\nuse it?"]
    tools = ["cosine · residual ratio · norm ratio", "retrieval probes\n(question / answer / graph)",
             "MEAN · SWAP · GSWAP\nwith and without text"]
    cols = [RED, YELLOW, GREEN]
    steps = VGroup()
    for i in range(3):
        b = box(qs[i], cols[i], w=3.4, h=1.2, size=20, fill=0.12)
        tl = T(tools[i], size=14, color=MUTED).next_to(b, DOWN, buff=0.15)
        steps.add(VGroup(b, tl))
    steps.arrange(RIGHT, buff=0.9).shift(UP * 1.0)
    arrows = VGroup(*[Arrow(steps[i][0].get_right(), steps[i + 1][0].get_left(), buff=0.1, color=MUTED)
                      for i in range(2)])
    s.play(LaggedStart(FadeIn(steps[0]), GrowArrow(arrows[0]), FadeIn(steps[1]), GrowArrow(arrows[1]),
                       FadeIn(steps[2]), lag_ratio=0.35), run_time=2.2)

    # the five diagnoses, worded after Section 4.5; examples are Table 5 rows (Llama-2)
    diag = [("SOFT PROMPT", "collapsed baseline, no resolved graph-specific use", "G-Retriever, all 3 datasets", RED),
            ("PRIOR", "no-entity token carries Q & A, used under MEAN/SWAP,\nno resolved graph-specific use",
             "no entity features, WebQSP", YELLOW),
            ("DECODABLE", "graph recoverable, no resolved answer-level effect", "Ours, ExplaGraphs", "#C678DD"),
            ("MASKED", "GSWAP effect resolves only without the verbalised graph", "Ours, WebQSP", BLUE),
            ("USED", "GSWAP changes the score by > 2 points, p < .05", "Ours, SceneGraphs (−5.30)", GREEN)]
    rows = VGroup()
    for name, d, ex, c in diag:
        if name == "SOFT PROMPT":
            n = soft_prompt_badge(name, size=15, pad=0.12, height=0.42)
            n.stretch_to_fit_width(2.1)
        else:
            n = chip(name, c, size=15, h=0.42, fill=0.3)
            n[0].stretch_to_fit_width(2.1)
            n[1].move_to(n[0])
        dt = T(d, size=15, line_spacing=0.8)
        rows.add(VGroup(n, dt.next_to(n, RIGHT, buff=0.3)))
    rows.arrange(DOWN, aligned_edge=LEFT, buff=0.14).next_to(steps, DOWN, buff=0.45)
    rows.to_edge(LEFT, buff=0.7)
    ex_x = rows.get_left()[0] + 9.1
    for r, (_, _, ex, _) in zip(rows, diag):
        r.add(T(ex, size=14, color=MUTED).move_to([ex_x, r[0].get_y(), 0], aligned_edge=LEFT))
    s.play(LaggedStart(*[FadeIn(r, shift=0.15 * UP) for r in rows], lag_ratio=0.3), run_time=2.6)
    flash = rows[0][0][1].copy().set_fill(opacity=0).set_stroke(color=WHITE, width=4)
    s.play(ShowPassingFlash(flash, time_width=0.35), run_time=0.9)
    s.wait(3.0)
    clear(s)

    final = VGroup(
        T("Test graph-token alignment.", size=40, weight=BOLD),
        T("Don't infer it from task accuracy.", size=40, weight=BOLD, color=YELLOW),
    ).arrange(DOWN, buff=0.25).shift(UP * 0.7)
    link = T("github.com/difra100/Token_Collapse_KGQA", size=22, color=BLUE).next_to(final, DOWN, buff=0.8)
    arx = T("arXiv:2610.31572", size=20, color=MUTED).next_to(link, DOWN, buff=0.2)
    s.play(Write(final[0]), run_time=1.0)
    s.play(Write(final[1]), run_time=1.0)
    s.play(FadeIn(link), FadeIn(arx), run_time=0.8)
    s.wait(3)
    clear(s, run_time=1.0)


# ============================================================ teaser: later methods
def act_beyond(s: Scene, number="", details=False):
    """Appendix F, Tables 9-10: the cross-method audit on WebQSP."""
    head = section_title(s, number, "Not just G-Retriever")
    sub = T("Later graph soft prompting methods on WebQSP  ·  inter-sample cosine of mean-pooled tokens",
            size=16, color=MUTED).next_to(head, DOWN, buff=0.2)
    s.play(FadeIn(sub), run_time=0.5)

    # (method, variant, tokens per graph K, projector, inter-sample cosine) from Tables 9 and 10
    rows = [("GRAG-PCST", "query-conditioned GAT", "K = 1", "Sigmoid", "0.9992"),
            ("GRETPool", "TopK", "K ≈ 7.8", "Sigmoid", "1.0000"),
            ("GRETPool", "SAG", "K ≈ 7.8", "Sigmoid", "1.0000"),
            ("GRETPool", "DiffPool", "K = 8", "Sigmoid", "0.9997"),
            ("GRETPool", "DiffPool", "K = 100", "Sigmoid", "1.0000"),
            ("GraSP-PCST", "one token per node", "K ≈ 16.7", "Tanh", "0.9978")]
    rng = np.random.default_rng(9)
    cards, clouds, vals = VGroup(), [], VGroup()
    centers = [np.array([x, y, 0]) for y in (0.95, -1.5) for x in (-4.45, 0.0, 4.45)]
    for i, (name, var, k, prj, cos) in enumerate(rows):
        frame = RoundedRectangle(corner_radius=0.15, width=4.2, height=2.25, stroke_color=FAINT,
                                 fill_color="#161B22", fill_opacity=1).move_to(centers[i])
        well = Square(1.55, stroke_color=FAINT, stroke_width=1.5).move_to(frame.get_left() + RIGHT * 1.0)
        pts = np.clip(rng.normal(0, 0.28, (36, 2)), -0.68, 0.68)
        cl = VGroup(*[Dot(well.get_center() + np.array([*p, 0]), radius=0.04, color=PALETTE[j % 7])
                      for j, p in enumerate(pts)])
        txt = VGroup(T(name, size=19, weight=BOLD),
                     T(var, size=15, color=FG),
                     T(f"{k}  ·  {prj}", size=14, color=MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        txt.next_to(well, RIGHT, buff=0.25).align_to(well, UP)
        if txt.get_right()[0] > frame.get_right()[0] - 0.12:
            txt.scale_to_fit_width(frame.get_right()[0] - 0.12 - txt.get_left()[0], about_edge=LEFT)
        val = T(f"cos {cos}", size=22, weight=BOLD, color=RED).next_to(txt, DOWN, buff=0.22).align_to(txt, LEFT)
        cards.add(VGroup(frame, well, txt))
        clouds.append(cl)
        vals.add(val)
    grid = VGroup(*[VGroup(c, cl, v) for c, cl, v in zip(cards, clouds, vals)])
    schem = T("schematic points", size=13, color=MUTED).next_to(grid, DOWN, buff=0.06).align_to(grid, RIGHT)

    s.play(LaggedStart(*[FadeIn(VGroup(c, cl), shift=0.15 * UP) for c, cl in zip(cards, clouds)],
                       lag_ratio=0.15), FadeIn(schem), run_time=1.8)
    s.wait(0.4)
    s.play(*[d.animate.move_to(cards[i][1].get_center() + 0.025 * np.array([*rng.normal(size=2), 0]))
             .set_color(RED) for i, cl in enumerate(clouds) for d in cl],
           run_time=1.8, rate_func=rate_functions.ease_in_out_cubic)
    s.play(LaggedStart(*[FadeIn(v, scale=1.2) for v in vals], lag_ratio=0.12), run_time=1.2)
    s.play(Indicate(grid[5][0][2][2], color=YELLOW), run_time=1.0)

    cap = caption("Collapse in all six configurations: 1 to 100 tokens, Sigmoid or Tanh projectors.",
                  size=22, color=YELLOW)
    s.play(FadeIn(cap), run_time=0.7)
    s.wait(2.5)
    if details:
        s.play(*[FadeOut(m) for m in s.mobjects if not any(m is x for x in head)], run_time=0.7)
        sub2 = T("Same interventions as for G-Retriever  ·  WebQSP Hit@1, all six configurations",
                 size=16, color=MUTED).next_to(head, DOWN, buff=0.2)
        items = [("With the verbalised graph", "MEAN · SWAP · GSWAP", "no resolved effect  (|Δ| ≤ 0.92 points)", FG),
                 ("Removing the verbalised graph", "NOTXT", "−25.6 to −43.2 points", RED),
                 ("Without the verbalised graph", "MEAN · SWAP · GSWAP", "only limited, inconsistent sensitivity", MUTED)]
        rows2 = VGroup()
        for lab, what, res, col in items:
            l = T(lab, size=17, color=MUTED)
            w = chip(what, ORANGE if what != "NOTXT" else RED, size=17, h=0.5, fill=0.2)
            r = T(res, size=24, weight=BOLD, color=col)
            rows2.add(VGroup(l, VGroup(w, r).arrange(RIGHT, buff=0.35)).arrange(DOWN, aligned_edge=LEFT, buff=0.12))
        rows2.arrange(DOWN, aligned_edge=LEFT, buff=0.45).next_to(sub2, DOWN, buff=0.6)
        note = T("Single-seed audit: it supports a recurrence claim,\n"
                 "not that every graph-prompting method collapses.", size=16, color=MUTED, line_spacing=0.9)
        note.next_to(rows2, DOWN, buff=0.5)
        s.play(FadeIn(sub2), run_time=0.5)
        s.play(LaggedStart(*[FadeIn(r, shift=0.15 * RIGHT) for r in rows2], lag_ratio=0.5), run_time=2.2)
        s.wait(1.2)
        s.play(FadeIn(note), run_time=0.6)
        s.wait(2.8)
    clear(s)


def act_teaser_end(s: Scene):
    q1 = T("So what does the collapsed token do?", size=38, weight=BOLD)
    q2 = T("And can collapse be prevented?", size=38, weight=BOLD)
    qs = VGroup(q1, q2).arrange(DOWN, buff=0.45).shift(UP * 1.3)
    s.play(Write(q1), run_time=1.2)
    s.wait(0.6)
    s.play(Write(q2), run_time=1.1)
    s.wait(0.8)

    cta = T("Read the paper and watch the full video to find out!", size=30, weight=BOLD)
    cta.set_color_by_gradient(*SP_COLORS)
    cta.next_to(qs, DOWN, buff=0.8)
    arx = T("arXiv:2610.31572", size=22, color=FG).next_to(cta, DOWN, buff=0.5)
    link = T("full video & code:  github.com/difra100/Token_Collapse_KGQA", size=22, color=BLUE)
    link.next_to(arx, DOWN, buff=0.2)
    s.play(FadeIn(cta, shift=0.2 * UP), run_time=0.9)
    flash = Underline(cta, color=SP_COLORS[1], stroke_width=4)
    s.play(ShowPassingFlash(flash, time_width=0.5), run_time=0.9)
    s.play(FadeIn(arx), FadeIn(link), run_time=0.8)
    s.wait(3.5)
    clear(s, run_time=1.0)



# ============================================================ scenes
class Title(SlowScene):
    def construct(self):
        act_title(self)


class Pipeline(SlowScene):
    def construct(self):
        act_pipeline(self)


class Collapse(SlowScene):
    def construct(self):
        act_collapse(self)


class Interventions(SlowScene):
    def construct(self):
        act_interventions(self)


class Causes(SlowScene):
    def construct(self):
        act_causes(self)


class TwoStage(SlowScene):
    def construct(self):
        act_two_stage(self)


class Takeaway(SlowScene):
    def construct(self):
        act_takeaway(self)


class FullVideo(SlowScene):
    def construct(self):
        act_title(self)
        act_pipeline(self)
        act_collapse(self, number="1 ·")
        act_interventions(self)
        act_beyond(self, number="3 ·", details=True)
        act_two_stage(self)
        act_takeaway(self)


class Beyond(SlowScene):
    def construct(self):
        act_beyond(self, number="3 ·", details=True)


class Teaser(SlowScene):
    """Short cut for social media: title, collapse, later methods, then the open questions."""
    def construct(self):
        act_title(self)
        act_pipeline(self)
        act_collapse(self, number="")
        act_beyond(self)
        act_teaser_end(self)
