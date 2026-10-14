"""Three-state provenance audit (logged evidence; Scorecard.auditability unchanged).

States per admitted skill:
  * ``auditable``   - provenance non-empty, every id is a train trace this run's learner
                      saw, none from probe/test (exactly the Scorecard.auditability rule);
  * ``external``    - provenance complete but from a DIFFERENT run/corpus: the learner
                      declared it external (``learner.external_provenance[skill_id]``), it
                      is non-empty, carries no placeholder marker, is disjoint from this
                      run's learn ids AND from probe/test ids;
  * ``unauditable`` - anything else (empty, placeholder, mixed, or held-out leak).

Scorecard.auditability still counts only ``auditable`` (canonical type + M0 unchanged);
the external state is reported via the ``audit`` log event and the analysis pipeline.
"""
from __future__ import annotations

ORIGIN_UNAVAILABLE = "origin_unavailable"
_MARKER_PREFIXES = ("frozen:", ORIGIN_UNAVAILABLE)


def _is_marker(pid: str) -> bool:
    return str(pid).startswith(_MARKER_PREFIXES)


def classify(skill, learner_ids, heldout_ids, external=None) -> str:
    prov = set(skill.provenance or [])
    if not prov or any(_is_marker(p) for p in prov) or prov & set(heldout_ids):
        return "unauditable"
    if prov <= set(learner_ids):
        return "auditable"
    if (external or {}).get(skill.id) is not None and not prov & set(learner_ids):
        return "external"
    return "unauditable"


def audit_states(skills, learner_ids, heldout_ids, external=None) -> dict:
    per = []
    for s in skills:
        st = classify(s, learner_ids, heldout_ids, external)
        rec = {"skill_id": s.id, "state": st, "n_provenance": len(set(s.provenance or []))}
        if st == "external":
            rec["external_provenance"] = True
            rec["origin"] = (external or {}).get(s.id)
        per.append(rec)
    n = len(per)
    c = {k: sum(1 for p in per if p["state"] == k) for k in ("auditable", "external", "unauditable")}
    return {"per_skill": per, "n_skills": n, **{f"n_{k}": v for k, v in c.items()},
            "auditability_local": (c["auditable"] / n) if n else 1.0,
            "traceable_fraction": ((c["auditable"] + c["external"]) / n) if n else 1.0}
