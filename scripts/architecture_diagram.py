"""Generate docs/architecture-{light,dark}.svg, the README architecture diagram.

Usage: python3 scripts/architecture_diagram.py [output_dir]   (default: docs)
"""
import sys
from pathlib import Path
from xml.sax.saxutils import escape

PALETTES = {
    "light": {
        "bg": "#ffffff", "text": "#1f2328", "muted": "#59636e", "line": "#d1d9e0", "arrow": "#59636e",
        "flow": "#cf222e", "panel": "#f6f8fa", "panel_stroke": "#d1d9e0",
        "client": "#ddf4ff", "client_s": "#54aeff",
        "api": "#ffffff", "api_s": "#afb8c1",
        "host": "#fbefff", "host_s": "#c297ff",
        "mem": "#fff8c5", "mem_s": "#d4a72c",
        "gpu": "#dafbe1", "gpu_s": "#4ac26b",
        "badge": "#cf222e", "badge_text": "#ffffff",
    },
    "dark": {
        "bg": "#0d1117", "text": "#e6edf3", "muted": "#9198a1", "line": "#30363d", "arrow": "#9198a1",
        "flow": "#ff7b72", "panel": "#151b23", "panel_stroke": "#30363d",
        "client": "#0c2d6b", "client_s": "#4493f8",
        "api": "#0d1117", "api_s": "#3d444d",
        "host": "#271843", "host_s": "#a371f7",
        "mem": "#3b2300", "mem_s": "#d29922",
        "gpu": "#0f2d1a", "gpu_s": "#3fb950",
        "badge": "#ff7b72", "badge_text": "#0d1117",
    },
}

W, H = 1280, 940
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
out: list[str] = []


def text(x, y, s, size=12.5, weight=400, fill="text", anchor="start", mono=False):
    family = MONO if mono else SANS
    out.append(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
               f'fill="{{{fill}}}" text-anchor="{anchor}">{escape(s)}</text>')


def rect(x, y, w, h, fill, stroke, r=10, dash=False, sw=1.5):
    d = ' stroke-dasharray="5 4"' if dash else ""
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{{{fill}}}" stroke="{{{stroke}}}" stroke-width="{sw}"{d}/>')


def box(x, y, w, h, title, lines, fill, stroke, mono_lines=(), badge=None):
    rect(x, y, w, h, fill, stroke)
    tx = x + 16
    if badge:
        out.append(f'<circle cx="{x + 24}" cy="{y + 22}" r="11" fill="{{badge}}"/>')
        text(x + 24, y + 26.5, badge, 12.5, 700, "badge_text", "middle")
        tx = x + 44
    text(tx, y + 27, title, 15, 600)
    for i, line in enumerate(lines):
        text(x + 16, y + 50 + i * 19, line, 12.5, 400, "muted", mono=i in mono_lines)


def arrow(x1, y1, x2, y2, color="arrow", label=None, lx=None, ly=None, anchor="start", both=False, dash=False, sw=1.6):
    marker = "flowhead" if color == "flow" else "head"
    start = f' marker-start="url(#{marker}-rev)"' if both else ""
    d = ' stroke-dasharray="5 4"' if dash else ""
    out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{{{color}}}" stroke-width="{sw}"{d} marker-end="url(#{marker})"{start}/>')
    if label:
        text(lx if lx is not None else (x1 + x2) / 2 + 8, ly if ly is not None else (y1 + y2) / 2 + 4, label, 12, 500, color if color == "flow" else "muted", anchor)


labels: list[tuple] = []


def panel(x, y, w, h, label, sub):
    rect(x, y, w, h, "panel", "panel_stroke", 14)
    labels.append((x, y, label, sub))


def draw_labels():
    # Drawn last with a halo in the panel color, so arrows that cross a label pass behind it.
    for x, y, label, sub in labels:
        halo = 'stroke="{panel}" stroke-width="6" paint-order="stroke" stroke-linejoin="round"'
        out.append(f'<text x="{x + 18}" y="{y + 26}" font-family="{SANS}" font-size="11.5" fill="{{muted}}" {halo}>'
                   f'<tspan font-weight="700" letter-spacing="0.6">{escape(label.upper())}</tspan>'
                   f'<tspan dx="10">{escape(sub)}</tspan></text>')


# Title
text(40, 46, "HotShift architecture", 24, 700)
text(40, 70, "A bounded GPU cache in front of a CPU-resident embedding table: how one microbatch flows from HTTP to the GPU and back", 13.5, 400, "muted")

# Clients
box(40, 96, 270, 78, "Dashboard", ["React + Vite · polls status and metrics"], "client", "client_s")
box(330, 96, 270, 78, "HTTP clients", ["recommend · pool · open-loop load tests"], "client", "client_s")
box(970, 96, 270, 78, "Offline replay", ["bench / bench-cuda · calls the engine"], "client", "client_s")

# API process
panel(40, 200, 1200, 176, "API process", "FastAPI on one uvicorn worker")
box(60, 240, 250, 116, "FastAPI", ["Pydantic validation (422)", "429 queue full · 503 busy", "504 past deadline"], "api", "api_s")
box(330, 240, 270, 116, "Batch scheduler", ["bounded FIFO queue (1,024)", "≤ 64 requests · ≤ 16,384 IDs", "≤ 2 ms wait for new arrivals"], "api", "api_s")
box(620, 240, 300, 116, "Engine worker thread", ["sole owner of all cache and GPU state", "reset / policy switch drain first", "results resolved on the event loop"], "api", "api_s")
box(950, 240, 270, 116, "Metrics + scoring", ["5 s rolling hit rate, p95 / p99", "top-k over 512 candidates", "(NumPy, on the CPU)"], "api", "api_s")

arrow(175, 174, 175, 240, both=True)
arrow(465, 174, 465, 240, both=True)
arrow(310, 298, 330, 298)
arrow(600, 298, 620, 298)
arrow(920, 298, 950, 298)
# Offline replay bypasses HTTP: route it down the right margin into the engine panel.
out.append('<polyline points="1240,135 1262,135 1262,430 1244,430" fill="none" stroke="{arrow}" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#head)"/>')

# Host engine
panel(40, 402, 1200, 236, "CUDACacheEngine · host", "Python + NumPy, pinned staging")
box(60, 438, 470, 58, "Embedding table  N × D FP32", [], "mem", "mem_s")
text(76, 486, "memory-mapped, read-only · 1M × 128 = 512 MB", 12.5, 400, "muted")
box(560, 438, 660, 58, "Pinned staging buffers", [], "mem", "mem_s")
text(576, 486, "unique IDs, miss rows, descriptors, offsets, outputs · reused after their copy event", 12.5, 400, "muted")

box(60, 520, 260, 100, "Deduplicate", ["np.unique → sorted unique IDs", "+ inverse index per request"], "host", "host_s", badge="1")
box(340, 520, 260, 100, "Pack misses", ["copy miss rows from the table", "build slot / miss descriptors"], "host", "host_s", badge="2")
box(620, 520, 290, 100, "Policy decides", ["static · lru · decayed_lfu · adaptive", "returns admissions + evictions"], "host", "host_s", badge="4")
box(930, 520, 290, 100, "Maintenance + CPU mirror", ["validate the delta, then publish", "slot_to_item, free-slot stack"], "host", "host_s", badge="5")
arrow(470, 496, 470, 520)
arrow(910, 570, 930, 570)
arrow(770, 356, 770, 402, label="engine.pool(histories)", lx=782, ly=385)

# PCIe
y_bus = 664
out.append(f'<line x1="40" y1="{y_bus}" x2="1240" y2="{y_bus}" stroke="{{line}}" stroke-width="2" stroke-dasharray="2 6" stroke-linecap="round"/>')
text(40, y_bus - 8, "PCIe", 11.5, 700, "muted")

# GPU
panel(40, 690, 1200, 200, "GPU", "one CUDA stream, custom kernels in csrc/")
box(60, 726, 260, 82, "probe_slots", ["map each unique ID to a slot or -1"], "gpu", "gpu_s")
box(340, 726, 570, 82, "pool_mean", ["mean of each request's rows, read from cached slots and staged misses"], "gpu", "gpu_s", badge="3")
box(930, 726, 290, 82, "scatter_rows_ + update_mapping_", ["invalidate → copy rows → publish"], "gpu", "gpu_s")
box(60, 826, 260, 50, "item_to_slot[N] · int32 · 4 MB", [], "mem", "mem_s")
box(340, 826, 880, 50, "cache_rows[C × D] · FP32 · 51 MB (10% of rows) + device miss staging", [], "mem", "mem_s")

# Host <-> device flow
arrow(150, 620, 150, 726, "flow", "unique IDs", 160, 680)
arrow(240, 726, 240, 620, "flow", "slots ⏸", 250, 652)
arrow(430, 620, 430, 726, "flow", "miss rows", 440, 680)
arrow(800, 726, 800, 620, "flow", "pooled vectors ⏸ → callers", 810, 652)
arrow(1075, 620, 1075, 726, "flow", "admissions ⏸", 1085, 680)
arrow(190, 808, 190, 826, both=True, sw=1.2)
arrow(625, 808, 625, 826, both=True, sw=1.2)
arrow(1075, 808, 1075, 826, sw=1.2)

draw_labels()

# Legend
text(40, 922, "1–5 run in order for every microbatch; ⏸ marks a host wait on a CUDA event (probe_done, output_done, maintenance_done). "
     "The pipeline is serialized: no host planning overlaps device work.", 12, 400, "muted")


def render(theme: str) -> str:
    pal = PALETTES[theme]
    body = "\n".join(out).format(**pal)
    defs = (
        "<defs>"
        f'<marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{pal["arrow"]}"/></marker>'
        f'<marker id="head-rev" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M10,0 L0,5 L10,10 z" fill="{pal["arrow"]}"/></marker>'
        f'<marker id="flowhead" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{pal["flow"]}"/></marker>'
        f'<marker id="flowhead-rev" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M10,0 L0,5 L10,10 z" fill="{pal["flow"]}"/></marker>'
        "</defs>"
    )
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
            f'aria-labelledby="t d"><title id="t">HotShift architecture</title>'
            f'<desc id="d">Clients call a FastAPI process whose batch scheduler feeds a single engine worker thread. '
            f'The host engine deduplicates IDs, probes the GPU slot map, packs misses from a memory-mapped CPU table, '
            f'pools on the GPU, lets the policy choose admissions and evictions, and applies them with scatter and map-update kernels.</desc>'
            f'{defs}<rect width="{W}" height="{H}" fill="{pal["bg"]}"/>\n{body}\n</svg>\n')


dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "docs"
for theme in PALETTES:
    (dest / f"architecture-{theme}.svg").write_text(render(theme), encoding="utf-8")
print("written")
