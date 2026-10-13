"""OfficeQAAdapterV2 (`benchmark: officeqa_v2`): OfficeQAAdapter with better retrieval.

* Date-aware narrowing: parse years / month-year mentions from the question and
  restrict search to treasury_bulletin_YYYY_MM files for those dates (year Y and
  Y+1, since later bulletins carry revised full-year figures; an explicit
  "<Month> YYYY" maps to that exact issue). Falls back to the global corpus.
* Corpus is the union of those date-candidate files (+ gold source files), not
  just the gold files, so retrieval is not oracle-restricted.
* Table-aware chunking: split on 'Table ...' title lines; a table stays whole
  (title + header rows) and is only row-split, with title+header repeated, when
  larger than OFFICEQA_V2_MAX_CHUNK chars (default 6000).
* Retrieve up to OFFICEQA_V2_K chunks (default 20) within a char budget
  OFFICEQA_V2_CTX_CHARS (default 24000).
* Tolerance default 0.01 (OfficeQA README), overridable by OFFICEQA_TOLERANCE.
"""
from __future__ import annotations
import os
import re
from pathlib import Path
from harness.benchmarks.officeqa import OfficeQAAdapter, BM25, HF_REPO, HF_PARSED_DIR
from harness.benchmarks import _pinned
from harness.registry import register

MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"])}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})
FILE_RE = re.compile(r"treasury_bulletin_(\d{4})_(\d{2})\.txt$")


def date_candidates(question: str, available: list[str], max_years: int = 4) -> list[str]:
    q = question.lower()
    exact = set()
    for m, y in re.findall(r"\b(" + "|".join(MONTHS) + r")[a-z]*\.?\s+(\d{4})\b", q):
        exact.add(f"treasury_bulletin_{y}_{MONTHS[m]:02d}.txt")
    years = sorted({int(y) for y in re.findall(r"\b(19[3-9]\d|20[0-2]\d)\b", q)})
    if len(years) > max_years:
        years = [years[0], years[-1]]
    want = set()
    for y in years:
        want |= {y, y + 1}
    avail = set(available)
    cands = {f for f in avail if (m := FILE_RE.search(f)) and int(m.group(1)) in want}
    if "bulletin" in q and exact & avail:
        return sorted(exact & avail)
    return sorted(cands | (exact & avail))


def table_chunks(text: str, max_chars: int) -> list[str]:
    lines = text.splitlines()
    blocks, cur = [], []
    for ln in lines:
        if re.match(r"^\s*Table\s+[A-Z]", ln) and cur:
            blocks.append(cur); cur = []
        cur.append(ln)
    if cur:
        blocks.append(cur)
    out = []
    for b in blocks:
        s = "\n".join(b)
        if len(s) <= max_chars:
            out.append(s); continue
        tbl = [i for i, ln in enumerate(b) if ln.startswith("|")]
        head_end = (tbl[1] + 1) if len(tbl) > 1 else min(len(b), 3)
        header = b[:head_end]
        hdr_txt = "\n".join(header)
        piece = list(header)
        for ln in b[head_end:]:
            if len("\n".join(piece)) + len(ln) > max_chars and len(piece) > len(header):
                out.append("\n".join(piece)); piece = list(header) + ["(continued)"]
            piece.append(ln[:max_chars - len(hdr_txt) - 20])
        out.append("\n".join(piece))
    return out


@register("benchmark", "officeqa_v2")
class OfficeQAAdapterV2(OfficeQAAdapter):
    def __init__(self, *a, **kw):
        if kw.get("tolerance") is None and "OFFICEQA_TOLERANCE" not in os.environ:
            kw["tolerance"] = 0.01
        self.k = int(os.environ.get("OFFICEQA_V2_K", 20))
        self.ctx_chars = int(os.environ.get("OFFICEQA_V2_CTX_CHARS", 24000))
        self.max_chunk = int(os.environ.get("OFFICEQA_V2_MAX_CHUNK", 6000))
        super().__init__(*a, **kw)

    def _load_corpus(self, rows):
        if self.source == "sample":
            return super()._load_corpus(rows)
        from huggingface_hub import list_repo_files, snapshot_download
        tok = os.environ.get("HF_TOKEN")
        allf = [f.split("/")[-1] for f in list_repo_files(HF_REPO, repo_type="dataset", token=tok)
                if f.startswith(HF_PARSED_DIR + "/") and f.endswith(".txt")]
        need = set()
        for r in rows:
            need |= set(date_candidates(r["question"], allf))
            need |= {n.strip() for n in re.split(r"[,\n;]", r["source_files"]) if n.strip()}
        out = _pinned.CACHE / "officeqa-corpus"
        snapshot_download(HF_REPO, repo_type="dataset", token=tok, local_dir=out,
                          allow_patterns=[f"{HF_PARSED_DIR}/{n}" for n in sorted(need)])
        self.corpus_files = sorted(need)
        return out / HF_PARSED_DIR

    def _chunk_corpus(self):
        files = getattr(self, "corpus_files", None) or sorted(p.name for p in Path(self.corpus_dir).glob("*.txt"))
        chunks, src = [], []
        for name in files:
            p = Path(self.corpus_dir) / name
            if not p.exists():
                continue
            for c in table_chunks(p.read_text(errors="ignore"), self.max_chunk):
                if c.strip():
                    chunks.append(c); src.append(name)
        self._by_file: dict[str, list[int]] = {}
        for i, s in enumerate(src):
            self._by_file.setdefault(s, []).append(i)
        return chunks, src

    def rank(self, question: str) -> list[int]:
        """Chunk indices, date-narrowed when possible, else global BM25."""
        cands = date_candidates(question, list(self._by_file))
        idx = [i for f in cands for i in self._by_file.get(f, [])]
        if idx:
            local = BM25([self._chunks[i] for i in idx])
            return [idx[j] for j in local.top(question, len(idx))]
        return self._bm25.top(question, len(self._chunks))

    def _render(self, r: dict) -> str:
        picked, used = [], 0
        for i in self.rank(r["question"])[: self.k]:
            c = self._chunks[i]
            if used + len(c) > self.ctx_chars and picked:
                break
            picked.append(i); used += len(c)
        ctx = "\n\n".join(f"[{self._chunk_src[i]}]\n{self._chunks[i]}" for i in picked)
        return (f"Question: {r['question']}\n\nTreasury Bulletin excerpts (retrieved):\n{ctx}\n\n"
                f"Answer with the final value only.")
