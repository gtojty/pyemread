"""Create the figures of the article from the package code and the example data.

Run from the repository root after the validation scripts:

    python validation/carr_benchmark.py      # needs the Carr et al. repository
    python validation/simulations.py
    python validation/eyelink_event_check.py
    python paper/make_figures.py

Figures are written to paper/figures/.
"""

import logging
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pyemread import cal, ext, ext_generic as eg, gen, lines  # noqa: E402

logging.getLogger("pyemread").setLevel(logging.ERROR)
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = ROOT / "validation" / "results"
EX = ROOT / "examples"
REGIONS = ["story01.region.csv", "story02.region.csv", "story03.region.csv"]
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "savefig.dpi": 300})


def label(ax, s):
    ax.text(-0.02, 1.02, s, transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="bottom", ha="right")


# ---------------------------------------------------------------------------
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.set_xlim(0, 30); ax.set_ylim(0, 23.5); ax.axis("off")
    col = {"input": "#e8e8e8", "stim": "#cfe0f1", "extract": "#fff1cc",
           "assign": "#f4d3d3", "measure": "#d9ead3", "output": "#fde29b"}

    def box(x, y, w, h, kind, title, body=""):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.3",
                                    fc=col[kind], ec="black", lw=0.8))
        ax.text(x + w / 2, y + h - 0.45, title, ha="center", va="top", fontsize=7.6,
                fontweight="bold")
        if body:
            ax.text(x + w / 2, y + h - 1.35, body, ha="center", va="top", fontsize=6.6,
                    linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color="0.25"))

    # inputs
    box(0.3, 19.6, 8.6, 3.3, "input", "Passage text", "lines of a text,\nfont, screen layout")
    box(10.7, 19.6, 8.6, 3.3, "input", "EyeLink recording", "ASC file\n(edf2asc)")
    box(21.1, 19.6, 8.6, 3.3, "input", "Other recordings",
        "fixation table, Eye-Tracking-BIDS,\nraw samples; Tobii, Pupil Labs\n(experimental)")
    # modules
    box(0.3, 12.2, 8.6, 6.0, "stim", "gen (stimuli)",
        "praster(): bitmap and\nword region file\n\ndraw_sac_fix(),\ndraw_line_assignment(),\nanimate()")
    box(10.7, 13.7, 8.6, 4.5, "extract", "ext (eyelink)",
        "process_subject():\nread ASC, assign lines\nand words, write tables")
    box(21.1, 13.7, 8.6, 4.5, "extract", "ext_generic (gaze)",
        "read_fixations(),\nread_bids_eyetrack(),\ndetect_events() (I-VT, I-DT),\nepoch(), process_fixations()")
    box(10.7, 8.4, 19.0, 4.2, "assign", "lines and ext_robust (diagnostics)",
        "fixation-to-line assignment: slice (default), diff (original heuristic), nearest,\n"
        "or line numbers corrected elsewhere; word assignment\n"
        "trial_report(): flags trials on which slice and diff disagree")
    box(10.7, 3.6, 19.0, 3.6, "measure", "cal (measures)",
        "cal_write_em(): 20 word-level and whole-trial measures\ncal_write_sm(): sentence-level measures")
    # outputs
    box(0.3, 3.6, 8.6, 6.8, "output", "Outputs",
        "stimulus bitmap (.png)\nregion file (.csv)\n\nfixation and saccade\ntables (.csv)\n\nplots, animations (.mp4)")
    box(10.7, 0.2, 19.0, 2.4, "output", "Per-word table (.csv) for mixed-effects analysis")
    arrow(4.6, 19.6, 4.6, 18.2)
    arrow(15.0, 19.6, 15.0, 18.2)
    arrow(25.4, 19.6, 25.4, 18.2)
    arrow(15.0, 13.7, 15.0, 12.6)
    arrow(25.4, 13.7, 25.4, 12.6)
    arrow(20.2, 8.4, 20.2, 7.2)
    arrow(20.2, 3.6, 20.2, 2.6)
    arrow(4.6, 12.2, 4.6, 10.4)
    arrow(8.9, 13.0, 10.7, 10.5)   # region file feeds assignment
    ax.text(9.05, 11.0, "region\nfile", fontsize=6.2, ha="left", va="center", color="0.25")
    fig.savefig(FIG / "fig1_architecture.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig2_stimulus(tmp):
    from matplotlib import font_manager
    font = font_manager.findfont("DejaVu Sans Mono")
    text = [ln.strip() for ln in open(EX / "textRasters" / "story01.txt", encoding="utf-8")
            if ln.strip() and not ln.startswith("#")]
    gen.praster(direct=str(tmp), fontpath=font, st_pos="TopLeft", lang_type="English",
                text=text, ID="story01")
    reg = pd.read_csv(tmp / "story01.region.csv")
    img = Image.open(tmp / "story01.png")
    x0, x1 = reg.x1_pos.min() - 25, reg.x2_pos.max() + 25
    y0, y1 = reg.y1_pos.min() - 25, reg.y2_pos.max() + 25
    fig = plt.figure(figsize=(7.2, 9.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.45, 0.42, 0.42], hspace=0.16)

    def draw(ax, xa, xb, ya, yb, boxes):
        ax.imshow(img, cmap="gray")
        if boxes:
            for r in reg.itertuples():
                ax.add_patch(Rectangle((r.x1_pos, r.y1_pos), r.x2_pos - r.x1_pos,
                                       r.y2_pos - r.y1_pos, fill=False, lw=0.7, ec="#d62728"))
        ax.set_xlim(xa, xb); ax.set_ylim(yb, ya); ax.axis("off")

    ax = fig.add_subplot(gs[0]); draw(ax, x0, x1, y0, y1, False); label(ax, "(a)")
    sel = reg[reg.line_no <= 2]
    ax = fig.add_subplot(gs[1])
    draw(ax, x0, sel.x2_pos.max() + 25, sel.y1_pos.min() - 20, sel.y2_pos.max() + 20, True)
    label(ax, "(b)")
    ax = fig.add_subplot(gs[2]); ax.axis("off"); label(ax, "(c)")
    cols = ["WordID", "Word", "length", "line_no", "x1_pos", "y1_pos", "x2_pos", "y2_pos"]
    t = ax.table(cellText=reg[cols].head(6).astype(str).values, colLabels=cols, loc="center",
                 cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.25)
    fig.savefig(FIG / "fig3_stimulus.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def process_examples(tmp):
    data = tmp / "data"
    data.mkdir()
    for r in REGIONS:
        shutil.copy(EX / "oralReading" / r, data / r)
    for s in ["1950138", "1950149"]:
        (data / s).mkdir()
        shutil.copy(EX / "oralReading" / s / f"{s}.asc", data / s)
        ext.process_subject(str(data), s, REGIONS)
    return data


def fig3_inspection(data):
    regions = ext.load_regions(str(data), REGIONS)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6))
    cmap = plt.get_cmap("tab10")
    for row, (s, tr, eye) in enumerate([("1950138", 1, "R"), ("1950149", 2, "L")]):
        fix = pd.read_csv(data / s / f"{s}_Fix.csv")
        f = fix[(fix.trial_id == tr) & (fix.eye == eye) & (fix.valid == "yes")].reset_index(drop=True)
        reg = regions[str(f.trial_type.iloc[0])]
        rep = lines.line_assignment_report(f, reg)
        for c, m in enumerate(["slice", "diff"]):
            ax = axes[row, c]
            res = lines.assign_lines(f, reg, method=m)
            for r in reg.itertuples():
                ax.add_patch(Rectangle((r.x1_pos, r.y1_pos), r.x2_pos - r.x1_pos,
                                       r.y2_pos - r.y1_pos, fill=False, lw=0.3, ec="0.65"))
            ax.plot(res.x_pos, res.y_pos, lw=0.3, color="0.55")
            ln = res.line_no.to_numpy(float); ok = np.isfinite(ln)
            ax.scatter(res.x_pos[ok], res.y_pos[ok], s=7, c=[cmap(int(v - 1) % 10) for v in ln[ok]],
                       zorder=3, linewidths=0)
            ax.scatter(res.x_pos[~ok], res.y_pos[~ok], s=9, marker="x", c="k", zorder=3, linewidths=0.6)
            ax.set_xlim(reg.x1_pos.min() - 60, reg.x2_pos.max() + 60)
            ax.set_ylim(reg.y2_pos.max() + 70, reg.y1_pos.min() - 70)
            ax.set_aspect("equal"); ax.tick_params(labelsize=7)
            ax.set_title(f"Participant {s}, trial {tr} ({eye} eye): {m}", fontsize=8)
            label(ax, "(" + "abcd"[2 * row + c] + ")")
        print(f"Fig 3 row {row}: agreement slice/diff = {rep['agree_slice_diff']:.2f}")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_inspection.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig4_events(tmp):
    s = "1950138"
    d = tmp / "ev"; (d / s).mkdir(parents=True)
    shutil.copy(EX / "oralReading" / s / f"{s}.asc", d / s)
    st = ext.read_timestamp(str(d), s, "RP")
    _, efix = ext.read_srr_asc(str(d), s, "RP", rec_last_fix=True, lump_fix=False)
    ppd = 40.5
    smp = eg.read_samples(st, "time", "x_pos2", "y_pos2", trial_col="trial_id", eye="R")
    smp = smp[smp.trial_id == 1]
    _, det = eg.detect_events(smp, method="IVT", velocity_threshold=30, px_per_deg=ppd)
    ref = efix[(efix.trial_id == 1) & (efix.eye == "R")]
    t0 = smp.time.iloc[0] + 9000
    w = smp[(smp.time >= t0) & (smp.time <= t0 + 3000)]
    t = (w.time - t0).to_numpy()
    dt = 4.0
    vx = eg._sg_derivative(w.x.to_numpy(float), 2) / dt * 1000 / ppd
    vy = eg._sg_derivative(w.y.to_numpy(float), 2) / dt * 1000 / ppd
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True,
                               gridspec_kw={"height_ratios": [1.4, 1]})
    a.plot(t, w.x, lw=0.8, color="k")
    ylo, yhi = w.x.min() - 40, w.x.max() + 40
    for df, y, colr, name in ((ref, yhi, "#1f77b4", "EyeLink parser"), (det, ylo, "#d62728", "pyemread I-VT")):
        sel = df[(df.end_time > t0) & (df.start_time < t0 + 3000)]
        for r in sel.itertuples():
            a.add_patch(Rectangle((r.start_time - t0, y - 18), r.end_time - r.start_time, 18 if y == yhi else -18,
                                  color=colr, alpha=0.8, lw=0))
        a.text(3010, y - 9, name, color=colr, va="center", fontsize=7)
    a.set_ylabel("Horizontal gaze position (px)"); a.set_ylim(ylo - 25, yhi + 25)
    label(a, "(a)")
    b.plot(t, np.hypot(vx, vy), lw=0.8, color="k")
    b.axhline(30, color="#d62728", ls="--", lw=0.8)
    b.set_ylabel("Velocity (°/s)"); b.set_xlabel("Time (ms)"); b.set_ylim(0, 400)
    label(b, "(b)")
    a.set_xlim(0, 3000)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_events.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig5_validation():
    acc = pd.read_csv(RES / "carr_accuracy.csv")
    acc = acc[acc.method != "slice"]                 # Carr et al.'s slice output ~ pyemread_slice
    order = (acc.groupby("method").acc_kept.mean().sort_values(ascending=False).index.tolist())
    names = {m: m.replace("pyemread_", "pyemread ") for m in order}
    sims = pd.read_csv(RES / "simulations.csv") if (RES / "simulations.csv").exists() else None
    fig = plt.figure(figsize=(7.2, 8.0))
    ax = fig.add_axes([0.10, 0.60, 0.88, 0.37])
    rng = np.random.default_rng(1)
    for i, m in enumerate(order):
        for j, (grp, colr) in enumerate((("adult", "#1f77b4"), ("child", "#ff7f0e"))):
            v = acc[(acc.method == m) & (acc.age_group == grp)].acc_kept.to_numpy()
            x = i + (j - 0.5) * 0.32 + rng.uniform(-0.07, 0.07, len(v))
            ax.scatter(x, v, s=6, color=colr, alpha=0.7, linewidths=0,
                       label={"adult": "adults", "child": "children"}[grp] if i == 0 else None)
            ax.plot([i + (j - 0.5) * 0.32 - 0.13, i + (j - 0.5) * 0.32 + 0.13], [np.median(v)] * 2,
                    color="k", lw=1.2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([names[m] for m in order], rotation=45, ha="right")
    for lab in ax.get_xticklabels():
        if lab.get_text().startswith("pyemread"):
            lab.set_fontweight("bold")
    ax.set_ylabel("Fixations on the correct line (%)"); ax.set_ylim(-3, 103)
    ax.legend(loc="lower left", fontsize=7, frameon=False)
    label(ax, "(a)")
    if sims is not None:
        sub = fig.add_gridspec(1, 5, wspace=0.15, left=0.10, right=0.98, top=0.40, bottom=0.10)
        show = ["pyemread_slice", "pyemread_diff", "attach", "warp", "cluster"]
        titles = {"noise": "Noise (SD, px)", "slope": "Slope", "shift": "Shift",
                  "regression_within": "P(within-line\nregression)",
                  "regression_between": "P(between-line\nregression)"}
        styles = {"pyemread_slice": ("#d62728", "-"), "pyemread_diff": ("#1f77b4", "-"),
                  "attach": ("0.5", ":"), "warp": ("#2ca02c", "--"), "cluster": ("#9467bd", "-.")}
        for k, f in enumerate(titles):
            a2 = fig.add_subplot(sub[k])
            g = sims[sims.factor == f].groupby(["method", "level"]).accuracy.mean().reset_index()
            for m in show:
                gm = g[g.method == m]
                a2.plot(gm.level, gm.accuracy, color=styles[m][0], ls=styles[m][1], lw=1.2,
                        label=names.get(m, m).replace("pyemread_", "pyemread "))
            a2.set_title(titles[f], fontsize=7.5); a2.set_ylim(0, 103); a2.tick_params(labelsize=6.5)
            if k:
                a2.set_yticklabels([])
            else:
                a2.set_ylabel("Accuracy (%)")
                label(a2, "(b)")
        handles, labs = a2.get_legend_handles_labels()
        fig.legend(handles, labs, loc="lower center", ncol=5, fontsize=7, frameon=False,
                   bbox_to_anchor=(0.54, 0.0))
    fig.savefig(FIG / "fig6_validation.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig6_measures():
    words = ["The", "old", "man", "sold", "his", "small", "boat", "today."]
    reg = pd.DataFrame({"WordID": range(1, 9), "Word": words, "length": [len(w) for w in words],
                        "line_no": 1, "x1_pos": [100 + 80 * k for k in range(8)],
                        "x2_pos": [180 + 80 * k for k in range(8)], "y1_pos": 90, "y2_pos": 110})
    seq = [1, 2, 4, 4, 3, 2, 4, 5, 6, 8, 3]            # word fixated by each fixation
    durs = [210, 230, 190, 160, 250, 200, 180, 220, 240, 260, 170]
    x = [140 + 80 * (w - 1) + (12 if k == 3 else 0) for k, w in enumerate(seq)]
    t = np.cumsum([0] + durs[:-1]) + 30 * np.arange(len(durs))
    fix = pd.DataFrame({"trial_id": 1, "trial_type": "s", "eye": "R", "start_time": t,
                        "end_time": t + durs, "duration": durs, "x_pos": x, "y_pos": 100.0,
                        "valid": "yes", "line_no": 1.0})
    fix = lines.assign_regions(fix, reg)
    em = cal.compute_word_measures(fix, reg).set_index("region")
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ys = np.arange(len(seq))[::-1]
    for k in range(8):
        ax.add_patch(Rectangle((k + 0.55, -0.9), 0.9, len(seq) + 0.4, color="0.94", lw=0))
        ax.text(k + 1, len(seq) - 0.1, words[k], ha="center", fontsize=8, fontweight="bold")
    ax.plot(seq, ys, color="0.55", lw=0.8)
    hl = {2: "#1f77b4", 3: "#1f77b4", 4: "#d62728", 5: "#d62728", 6: "#d62728"}
    for i, (w, y, d) in enumerate(zip(seq, ys, durs)):
        ax.scatter(w, y, s=d * 0.9, color=hl.get(i, "0.35"), zorder=3, edgecolor="k", lw=0.4)
        ax.text(w + 0.22, y, f"{i + 1}  ({d} ms)", va="center", fontsize=6.6)
    ax.set_yticks([]); ax.set_xticks([]); ax.set_xlim(0.4, 11.6); ax.set_ylim(-1.2, len(seq) + 0.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(0.45, -1.05, "fixation order (top to bottom); marker area proportional to duration",
            fontsize=6.5, color="0.3")
    w4, w3 = em.loc[4], em.loc[3]
    txt = ("Word 4 (sold):\n"
           f"  first-fixation duration = {w4.ffixurt:.0f} ms (fix. 3)\n"
           f"  gaze duration = {w4.fpurt:.0f} ms (fix. 3-4)\n"
           f"  regression-path duration = {w4.rpurt:.0f} ms (fix. 3-7)\n"
           f"  re-reading (second pass) = {w4.spurt:.0f} ms (fix. 7)\n"
           f"  total time = {w4.tfurt:.0f} ms\n\n"
           "Word 3 (man): fixated only after\n"
           "the eye had passed it, so it has\n"
           "no first-pass measures;\n"
           f"  second pass = {w3.spurt:.0f} ms (fix. 5, 11)")
    ax.text(8.75, len(seq) - 0.6, txt, fontsize=6.9, va="top", family="DejaVu Sans")
    fig.savefig(FIG / "fig2_measures.png", bbox_inches="tight")
    plt.close(fig)
    return em


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp())
    fig1_architecture()
    fig2_stimulus(tmp)
    data = process_examples(tmp)
    fig3_inspection(data)
    fig4_events(tmp)
    fig5_validation()
    em = fig6_measures()
    print(em.loc[[3, 4], ["ffixurt", "fpurt", "rpurt", "spurt", "tfurt"]])
    shutil.rmtree(tmp)
    print("Figures written to", FIG)
