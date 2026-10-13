r"""Render every experiment figure to PDF for inclusion in the paper.

tikzplotly names its pgfplots table columns after the plotly trace names, and
plotly express joins colour and line-dash levels with ", " -- the resulting
`table[y=Certified, 1.5]` is not valid pgfplots syntax, so the exported .tex
does not compile. We therefore re-render each figure from the JSON embedded in
the .html and write a PDF alongside it, which is what the notes \includegraphics.
"""

import json
import re
from pathlib import Path

import plotly.graph_objects as go

decoder = json.JSONDecoder()

# These scripts render their own multi-panel PDFs with a shared legend; re-rendering
# them here would flatten the layout.
SELF_RENDERED = {"iv_and_proxy", "distributional_failure"}

# Folders whose PDFs are written directly by figure_style.save() / paper_figures
# at a deliberate size; re-rendering here would overwrite them squashed.
STYLED_DIRS = {"iv_and_proxy", "distributional_failure"}
STYLED_PREFIX = "app_"  # river-discharge paper figures (hydro.experiments.paper_figures)

for html_path in sorted(Path("experiments").glob("*/*.html")):
    if html_path.parent.name in STYLED_DIRS or html_path.stem.startswith(STYLED_PREFIX):
        continue
    if html_path.parent.name in SELF_RENDERED:
        continue
    if not html_path.exists():  # another experiment may be rewriting its outputs
        continue
    html = html_path.read_text()
    match = re.search(r'Plotly\.newPlot\(\s*"[^"]+",\s*', html)
    if match is None:
        continue
    # Nested braces rule out a regex for the layout; decode the arguments in sequence.
    data, end = decoder.raw_decode(html, match.end())
    layout, _ = decoder.raw_decode(html, html.index("{", end))
    fig = go.Figure(data=data, layout=layout)
    # The LaTeX caption carries the title, so drop it. Where the figure left the y-axis
    # unlabelled (the simulation panels, whose y column is literally "value") the title is
    # the only description of what is plotted, so demote it rather than lose it; the river
    # panels label their own axes and their titles are far too long to sit beside one.
    # Sized so the fonts stay readable when the panel is included at half text width.
    # The dumbbell placebos carry one station-pair label per row and so are included at
    # full width in the notes instead; the same canvas then reads comfortably there.
    fig.update_layout(template="plotly_white", width=760, height=520, title=None, font_size=17)
    if (fig.layout.yaxis.title.text or "").strip() == "value":
        fig.update_yaxes(title_text=(layout.get("title") or {}).get("text") or "value")
    fig.update_xaxes(title_text=(fig.layout.xaxis.title.text or "").replace("_", " "))
    if fig.layout.legend.title.text:
        fig.update_layout(legend_title_text=fig.layout.legend.title.text.replace("_", " "))
    fig.write_image(html_path.with_suffix(".pdf"))
    print(f"wrote {html_path.with_suffix('.pdf')}")

print("Done")
