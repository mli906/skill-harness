"""Per-cell cost estimate for the scaled ALFWorld run (no LLM calls).
Token rates measured on M3/M3-Qwen logs: per step input ~731 (no-skill), ~900 (skill_md k=3),
~1815 (raw_trace k=3); output ~39/step. Episode = STEPS steps (6 task types, cap 30).
Cache: identical prompts hit (probe-empty + no-skill baseline shared across cells of the same
model/seed; NR & Bayesian test eval cached when they admit the same lib as noop -- not assumed)."""
import json, sys
STEPS = float(sys.argv[1]) if len(sys.argv) > 1 else 22
SEEDS, NT, NP = 3, 24, 24
PRICE = {"Q": (0.36, 0.40), "M": (0.15, 0.60)}            # OpenRouter $/M in,out
GPT4O = (2.5, 10.0)                                         # OpenAI direct (builds)
IN = {"none": 731, "md": 900, "rag": 1815}; OUT = 39


def ep(model, kind, n):  # $ for n episodes
    pi, po = PRICE[model]; return n * STEPS * (IN[kind] * pi + OUT * po) / 1e6


rows = []
def add(track, cell, model, kind, n_eps, note=""):
    rows.append({"track": track, "cell": cell, "model": model, "eps": n_eps, "usd_openrouter": round(ep(model, kind, n_eps), 3), "note": note})

# corpora (temp 0.7, k=2 x 24 learn tasks), once
add("corpus", "Q corpus (24x2)", "Q", "none", 48); add("corpus", "M corpus (24x2)", "M", "none", 48)
# Track A: consumer Q, Q corpus
add("A", "shared probe-empty + baseline", "Q", "none", SEEDS * (NP + NT))
for L, C, kind in [("trace2skill", 1, "md"), ("frozen_evoskill", 1, "md"), ("awm", 3, "md"), ("raw_trace_rag", 48, "rag")]:
    add("A", f"{L} x noop", "Q", kind, SEEDS * NT)
    add("A", f"{L} x non_regression", "Q", kind, SEEDS * (C * NP + NT), f"{C} candidates x 24 probe replays")
    add("A", f"{L} x bayesian", "Q", kind, SEEDS * (C * NP + NT), "assumes no cache overlap with NR")
add("A", "awm builds (1 call/seed, consumer model)", "Q", "none", 0.1 * SEEDS)
# Transfer 2x2 (trace2skill, noop): QQ = Track A cell (reused)
add("T", "M-corpus -> Q", "Q", "md", SEEDS * NT)
add("T", "Q-corpus -> M, M-corpus -> M", "M", "md", SEEDS * 2 * NT)
add("T", "M no-skill baseline", "M", "none", SEEDS * NT)
# Track B: EvoSkill C2, each model; prefilter 48 once; per seed 24 practice rollouts + probe/test(Q cached) + gates (C=2)
for m in "QM":
    add("B", f"{m} practice prefilter (48 x1)", m, "none", 48)
    add("B", f"{m} practice rollouts", m, "none", SEEDS * 24)
    if m == "M": add("B", "M probe-empty", m, "none", SEEDS * NP)
    add("B", f"{m} evoskill x noop", m, "md", SEEDS * NT)
    add("B", f"{m} evoskill x NR+bayes", m, "md", SEEDS * (2 * 2 * NP + 2 * NT), "2 skills assumed")
tot = {}
for r in rows: tot[r["track"]] = round(tot.get(r["track"], 0) + r["usd_openrouter"], 2)
openai = {"trace2skill builds (2 corpora x 3 seeds, ~80k tok gpt-4o)": round(6 * 80e3 * 3.0 / 1e6, 2),
          "evoskill builds (2 models x 3 seeds, ~$0.05)": 0.3}
out = {"steps_per_episode": STEPS, "rows": rows, "openrouter_by_track": tot,
       "openrouter_total": round(sum(tot.values()), 2), "openai_builds": openai,
       "openrouter_total_excl_rag_gating": round(sum(tot.values()) - sum(r["usd_openrouter"] for r in rows if "raw_trace_rag x " in r["cell"] and "noop" not in r["cell"]), 2)}
print(json.dumps(out, indent=1))
