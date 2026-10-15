"""Results-analysis pipeline (read-only over run logs).

    python -m harness.analysis --runs <dir> [--out <dir>] [--expected cells.json]

Parses per-run JSONL logs (run.start / split / train.traces / gate.result /
eval.task / run.end, plus the governance smoke's ``cell`` events) and governance
evidence logs, then writes a markdown report, CSV/JSON aggregates and Pareto
plots. Runs nothing, imports no run code, writes only under --out.
"""
from .parse import Run, load_runs, load_evidence  # noqa: F401
from .stats import mcnemar, paired_bootstrap, bh_adjust, holm_adjust  # noqa: F401
from .report import analyze, write_outputs  # noqa: F401
