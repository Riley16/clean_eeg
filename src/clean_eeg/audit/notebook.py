"""Audit report notebook builder + nbconvert executor.

The notebook is generated programmatically (no template file) so the
cell content lives next to the rest of the audit code and evolves with
it. Each subject audit produces three artifacts in the subject dir:

  - ``edf_audit.json`` (from the orchestrator; not this module)
  - ``edf_audit.ipynb`` — executed notebook
  - ``edf_audit.html`` — HTML render of the executed notebook

The notebook reads ``edf_audit.json`` from the subject dir at
execution time; it doesn't embed audit results directly so the same
notebook can be re-executed later against an updated JSON.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbconvert.exporters import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor


NOTEBOOK_FILENAME = "edf_audit.ipynb"
HTML_FILENAME = "edf_audit.html"


def _cell_load_audit(subject_dir: Path, audit_json_path: Path) -> str:
    return (
        "import json\n"
        "from pathlib import Path\n"
        f"SUBJECT_DIR = Path(r'''{subject_dir}''')\n"
        f"AUDIT_JSON_PATH = Path(r'''{audit_json_path}''')\n"
        "audit = json.loads(AUDIT_JSON_PATH.read_text())\n"
        "print(f\"Subject: {audit.get('subject_code', '?')}\")\n"
        "print(f\"Audit run at: {audit.get('generated_at', '?')}\")\n"
        "print(f\"Subject dir: {SUBJECT_DIR}\")"
    )


def _cell_summary_table() -> str:
    return (
        "from collections import Counter\n"
        "statuses = {name: r['status'] for name, r in audit['checks'].items()}\n"
        "counts = Counter(statuses.values())\n"
        "print(f\"pass: {counts.get('pass', 0)}   \"\n"
        "      f\"warn: {counts.get('warn', 0)}   \"\n"
        "      f\"fail: {counts.get('fail', 0)}\\n\")\n"
        "for name, status in statuses.items():\n"
        "    marker = {'pass': 'OK  ', 'warn': 'WARN', 'fail': 'FAIL'}[status]\n"
        "    print(f'  [{marker}] {name}')"
    )


def _cell_per_check_issues() -> str:
    return (
        "for name, r in audit['checks'].items():\n"
        "    issues = r.get('issues', [])\n"
        "    if not issues:\n"
        "        continue\n"
        "    print(f'--- {name} ({r[\"status\"]}) ---')\n"
        "    for msg in issues:\n"
        "        print(f'  {msg}')\n"
        "    print()"
    )


def _cell_annotation_matches() -> str:
    return (
        "scan = audit['checks'].get('annotation_phi_scan', {})\n"
        "matches = scan.get('matched_tokens', {})\n"
        "if not matches:\n"
        "    print('No name-dictionary matches in annotations.')\n"
        "else:\n"
        "    print(f'{len(matches)} token(s) matched the US-name dictionary:')\n"
        "    for token, hits in matches.items():\n"
        "        print(f\"  '{token}': {len(hits)} occurrence(s)\")\n"
        "        for h in hits[:5]:\n"
        "            print(f\"      {h['file']} @ {h['onset']}s: {h['text']!r}\")\n"
        "        if len(hits) > 5:\n"
        "            print(f'      ...and {len(hits) - 5} more')"
    )


def _cell_eeg_snippets(n_channel_plot: int, n_files_plot: int,
                       plot_seconds: float) -> str:
    return (
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from clean_eeg.audit.signals import read_signal_window\n"
        "from clean_eeg.audit.select import select_files\n"
        "\n"
        f"N_CHANNEL_PLOT = {n_channel_plot}\n"
        f"PLOT_SECONDS = {plot_seconds}\n"
        f"N_FILES_PLOT = {n_files_plot}\n"
        "\n"
        "recordings = sorted(SUBJECT_DIR.glob('*.edf'))\n"
        "recordings = [p for p in recordings if not p.name.endswith('_annotations.edf')]\n"
        "picks = select_files(recordings, n_files=N_FILES_PLOT)\n"
        "if not picks:\n"
        "    print('No recording files to plot.')\n"
        "else:\n"
        "    fig, axes = plt.subplots(len(picks), 1,\n"
        "                             figsize=(10, 2.2 * len(picks)),\n"
        "                             squeeze=False)\n"
        "    for ax, p in zip(axes[:, 0], picks):\n"
        "        window = read_signal_window(p, window_seconds=PLOT_SECONDS)\n"
        "        labels = list(window.keys())[:N_CHANNEL_PLOT]\n"
        "        for i, lab in enumerate(labels):\n"
        "            y = window[lab].astype(float)\n"
        "            y = (y - y.mean()) / (y.std() + 1e-9)\n"
        "            ax.plot(y + i * 4, linewidth=0.5)\n"
        "        ax.set_title(p.name, fontsize=9)\n"
        "        ax.set_yticks([i * 4 for i in range(len(labels))])\n"
        "        ax.set_yticklabels(labels, fontsize=8)\n"
        "        ax.set_xlabel('samples')\n"
        "    fig.tight_layout()\n"
        "    plt.show()"
    )


def build_audit_notebook(subject_dir: Path, audit_json_path: Path,
                         *,
                         n_channel_plot: int = 5,
                         n_files_plot: int = 4,
                         plot_seconds: float = 5.0,
                         ) -> nbf.NotebookNode:
    """Return the audit report notebook (unexecuted). Paths are baked
    into the first cell so the notebook can execute from any cwd.
    Plot params are also baked so the CLI's ``--n-channel-plot`` /
    ``--n-files-plot`` propagate all the way to the rendered figure.
    """
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    nb.cells = [
        nbf.v4.new_markdown_cell("# EDF audit report"),
        nbf.v4.new_code_cell(_cell_load_audit(subject_dir, audit_json_path)),
        nbf.v4.new_markdown_cell("## Summary"),
        nbf.v4.new_code_cell(_cell_summary_table()),
        nbf.v4.new_markdown_cell("## Per-check issues"),
        nbf.v4.new_code_cell(_cell_per_check_issues()),
        nbf.v4.new_markdown_cell("## Annotation PHI scan — dictionary hits"),
        nbf.v4.new_code_cell(_cell_annotation_matches()),
        nbf.v4.new_markdown_cell("## EEG snippet plots"),
        nbf.v4.new_code_cell(_cell_eeg_snippets(
            n_channel_plot=n_channel_plot,
            n_files_plot=n_files_plot,
            plot_seconds=plot_seconds,
        )),
    ]
    return nb


def render_audit_notebook(subject_dir: str | Path,
                          *,
                          output_dir: str | Path | None = None,
                          emit_html: bool = True,
                          timeout: int = 120,
                          n_channel_plot: int = 5,
                          n_files_plot: int = 4,
                          plot_seconds: float = 5.0,
                          ) -> tuple[Path, Path | None]:
    """Write ``edf_audit.ipynb`` + ``edf_audit.html`` into
    ``output_dir`` (defaults to ``subject_dir``). Executes the notebook
    in-process via ``nbconvert.ExecutePreprocessor``; requires
    ``ipykernel`` (declared in pyproject dependencies).
    """
    # Resolve to absolute paths — the executed notebook uses
    # ``output_dir`` as cwd (via ExecutePreprocessor's metadata.path),
    # so any relative path baked into a cell would resolve against
    # the wrong root at execution time.
    subject_dir = Path(subject_dir).resolve()
    output_dir = (Path(output_dir).resolve()
                  if output_dir is not None else subject_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_json_path = output_dir / "edf_audit.json"
    ipynb_path = output_dir / NOTEBOOK_FILENAME

    nb = build_audit_notebook(subject_dir, audit_json_path,
                              n_channel_plot=n_channel_plot,
                              n_files_plot=n_files_plot,
                              plot_seconds=plot_seconds)
    ExecutePreprocessor(timeout=timeout, kernel_name="python3").preprocess(
        nb, {"metadata": {"path": str(output_dir)}}
    )
    nbf.write(nb, str(ipynb_path))

    html_path: Path | None = None
    if emit_html:
        html_path = output_dir / HTML_FILENAME
        body, _ = HTMLExporter().from_notebook_node(nb)
        html_path.write_text(body)
    return ipynb_path, html_path
