"""OfficeQAAdapter: OfficeQA questions + Treasury Bulletin corpus, scored by
OfficeQA's own reward.score_answer (pinned databricks/officeqa).

Sources (env OFFICEQA_SOURCE):
  hf      -> databricks/officeqa on Hugging Face (gated; needs HF_TOKEN with access).
             CSV via `datasets`, parsed bulletins via hf_hub_download per source file.
  sample  -> 10-question sample + 9 bulletins from the pinned EvoSkill repo.
  auto    -> hf if HF_TOKEN set, else sample (default).
Other env: OFFICEQA_SPLIT (full|pro, default full), OFFICEQA_LIMIT (int),
OFFICEQA_TOLERANCE (float, default 0.0), OFFICEQA_TEST_FRAC (default 0.4),
OFFICEQA_CONTEXT_CHUNKS (default 3), OFFICEQA_CHUNK_CHARS (default 1500).

Each task string = question + top-k BM25-retrieved corpus chunks, so the fixed
ReActAgent (no file tools) can ground its answer. Category = difficulty.
"""
from __future__ import annotations
import csv
import hashlib
import math
import os
import re
from collections import Counter
from pathlib import Path
from harness.interfaces.benchmark import Benchmark
from harness.registry import register
from harness.benchmarks import _pinned

HF_REPO = "databricks/officeqa"
HF_PARSED_DIR = "treasury_bulletins_parsed/transformed"


def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [Counter(_tok(d)) for d in docs]
        self.len = [sum(c.values()) for c in self.docs]
        self.avg = sum(self.len) / max(1, len(self.len))
        df = Counter(t for c in self.docs for t in c)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def top(self, query: str, k: int) -> list[int]:
        q = _tok(query)
        def score(i):
            c, s = self.docs[i], 0.0
            for t in q:
                if t in c:
                    tf = c[t]
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            return s
        return sorted(range(len(self.docs)), key=lambda i: (-score(i), i))[:k]


def _bucket(uid: str) -> float:
    return int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


@register("benchmark", "officeqa")
class OfficeQAAdapter(Benchmark):
    def __init__(self, source: str | None = None, limit: int | None = None,
                 split: str | None = None, tolerance: float | None = None):
        env = os.environ
        source = source or env.get("OFFICEQA_SOURCE", "auto")
        if source == "auto":
            source = "hf" if env.get("HF_TOKEN") else "sample"
        self.source = source
        lim = limit if limit is not None else env.get("OFFICEQA_LIMIT")
        self.limit = int(lim) if lim not in (None, "", "0") else None
        self.split = split or env.get("OFFICEQA_SPLIT", "full")
        self.tolerance = float(tolerance if tolerance is not None else env.get("OFFICEQA_TOLERANCE", 0.0))
        self.n_chunks = int(env.get("OFFICEQA_CONTEXT_CHUNKS", 3))
        self.chunk_chars = int(env.get("OFFICEQA_CHUNK_CHARS", 1500))
        test_frac = float(env.get("OFFICEQA_TEST_FRAC", 0.4))
        self._reward = _pinned.officeqa_reward()

        rows = self._load_rows()
        rows.sort(key=lambda r: (_bucket(r["uid"]), r["uid"]))  # deterministic order
        if self.limit:
            rows = rows[: self.limit]
        self.rows = rows
        self.corpus_dir = self._load_corpus(rows)
        self._chunks, self._chunk_src = self._chunk_corpus()
        self._bm25 = BM25(self._chunks)

        self._by_task: dict[str, dict] = {}
        self._train, self._test = [], []
        for cat in sorted({r["difficulty"] for r in rows}):
            group = [r for r in rows if r["difficulty"] == cat]
            n_test = max(1, round(len(group) * test_frac)) if len(group) > 1 else 0
            for i, r in enumerate(group):
                t = self._render(r)
                self._by_task[t] = r
                (self._test if i < n_test else self._train).append(t)

    # ---- loading ------------------------------------------------------------
    def _load_rows(self) -> list[dict]:
        if self.source == "hf":
            from datasets import load_dataset
            ds = load_dataset(HF_REPO, data_files=f"officeqa_{self.split}.csv", split="train",
                              token=os.environ.get("HF_TOKEN"))
            return [dict(r) for r in ds]
        if self.source == "sample":
            with open(_pinned.evoskill_officeqa_sample() / "officeqa_sample.csv", newline="") as f:
                return list(csv.DictReader(f))
        raise ValueError(f"unknown OFFICEQA_SOURCE {self.source!r}")

    def _load_corpus(self, rows: list[dict]) -> Path:
        if self.source == "sample":
            return _pinned.evoskill_officeqa_sample() / "treasury_bulletins"
        from huggingface_hub import hf_hub_download
        out = _pinned.CACHE / "officeqa-corpus"
        names = sorted({n.strip() for r in rows for n in re.split(r"[,\n;]", r["source_files"]) if n.strip()})
        for n in names:
            hf_hub_download(HF_REPO, f"{HF_PARSED_DIR}/{n}", repo_type="dataset",
                            local_dir=out, token=os.environ.get("HF_TOKEN"))
        return out / HF_PARSED_DIR

    def _chunk_corpus(self) -> tuple[list[str], list[str]]:
        chunks, src = [], []
        for p in sorted(Path(self.corpus_dir).glob("*.txt")):
            text = p.read_text(errors="ignore")
            step = self.chunk_chars
            for i in range(0, len(text), step):
                chunks.append(text[i:i + step]); src.append(p.name)
        return chunks, src

    def _render(self, r: dict) -> str:
        idx = self._bm25.top(r["question"], self.n_chunks) if self._chunks else []
        ctx = "\n\n".join(f"[{self._chunk_src[i]}]\n{self._chunks[i]}" for i in idx)
        return (f"Question: {r['question']}\n\nTreasury Bulletin excerpts (retrieved):\n{ctx}\n\n"
                f"Answer with the final value only.")

    # ---- Benchmark ----------------------------------------------------------
    def train_tasks(self) -> list[str]:
        return list(self._train)

    def test_tasks(self) -> list[str]:
        return list(self._test)

    def verify(self, task: str, output: str) -> bool:
        return self._reward.score_answer(str(self._by_task[task]["answer"]), output, self.tolerance) >= 1.0

    # ---- optional duck-typed extensions -------------------------------------
    def category(self, task: str) -> str:
        return self._by_task[task]["difficulty"]

    def answer(self, task: str) -> str:
        return str(self._by_task[task]["answer"])

    def question(self, task: str) -> str:
        return self._by_task[task]["question"]

    def uid(self, task: str) -> str:
        return self._by_task[task]["uid"]
