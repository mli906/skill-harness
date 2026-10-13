from harness.benchmarks.officeqa_v2 import date_candidates, table_chunks

AV = [f"treasury_bulletin_{y}_{m:02d}.txt" for y in (1974, 1975, 1976, 1992) for m in range(1, 13)]


def test_date_candidates_year_and_exact_issue():
    c = date_candidates("For calendar year 1974, compute ...", AV)
    assert c and all(f[18:22] in ("1974", "1975") for f in c) and len(c) == 24
    assert date_candidates("listed in the U.S Treasury's June 1992 bulletin", AV) == ["treasury_bulletin_1992_06.txt"]
    assert date_candidates("no dates here", AV) == []


def test_table_chunks_keep_title_and_header():
    hdr = "Table FD-1. - Summary\n(In millions)\n| a | b |\n| --- | --- |"
    body = "\n".join(f"| {i} | {i*2} |" for i in range(400))
    text = "intro text\n" + hdr + "\n" + body + "\nTable FD-2. - Other\n| x |\n| --- |\n| 1 |"
    ch = table_chunks(text, 1500)
    assert ch[0].startswith("intro")
    fd1 = [c for c in ch if c.startswith("Table FD-1")]
    assert len(fd1) > 1 and all(c.startswith(hdr) for c in fd1)
    assert all(len(c) <= 1500 + 50 for c in fd1)
    assert any(c.startswith("Table FD-2") and "| 1 |" in c for c in ch)
