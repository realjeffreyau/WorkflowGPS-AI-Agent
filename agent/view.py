"""One JSON shape for the UI, built from a data root and a run folder.

Demo (ui/export.py) and Live (agent/workspace.py) both call build(), so every tab
renders the same way in either mode. Never reads data/eval.
"""
import json
import re
from pathlib import Path
from agent.store import source_items

# Display only: tickets whose title asks someone to approve or choose. The agent never sees this label.
DECISION = re.compile(r"^(Review|Design review|Approve|Decide|Choose)\b")


def _load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def ticket_view(t):
    return {"title": t.title, "goal": t.goal, "customer": t.customer, "owner": t.owner, "status": t.status,
            "priority": t.priority, "due": t.due.isoformat() if t.due else None, "estimate": t.estimate_hours,
            "description": t.description, "kind": "decision" if DECISION.match(t.title) else "action"}


def source_view(day):
    out = {}
    for key, item in source_items(day):
        if "-M" in key:
            out[key] = {"kind": "meeting", "who": item.attendees[0] if item.attendees else None, "time": item.time,
                        "label": item.title, "attendees": item.attendees, "text": "\n".join(item.transcript)}
        elif "-S" in key:
            out[key] = {"kind": "slack", "who": item.author, "time": item.time, "label": item.channel, "text": item.text}
        else:
            patch = item.set.model_dump(mode="json", exclude_unset=True)
            fields = ", ".join(f"{k}: {v}" for k, v in patch.items())
            out[key] = {"kind": "jira", "who": item.by, "time": None, "label": f"{item.action} {item.ticket}",
                        "ticket": item.ticket, "action": item.action, "text": "; ".join(x for x in (fields, item.comment) if x)}
    return out


def lesson_changes(before, after, n):
    old = {l["id"]: l for l in before}
    out = []
    for l in after:
        prev = old.get(l["id"])
        if prev and prev["rule"] == l["rule"] and prev.get("applies_to") == l.get("applies_to"):
            continue
        tag = f"D{n}:"
        out.append({"kind": "revised" if prev else "added", "id": l["id"], "rule": l["rule"],
                    "before": prev["rule"] if prev else None, "applies_to": l.get("applies_to", []),
                    "from": [e[len(tag):] for e in l.get("evidence", []) if e.startswith(tag)]})
    return out


def calls_summary(path, day):
    path = Path(path)
    if not path.exists():
        return None
    calls = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    calls = [c for c in calls if c.get("day") == day]
    if not calls:
        return None
    return {"total": len(calls), "cache_hits": sum(bool(c.get("cache_hit")) for c in calls),
            "retries": sum(c.get("attempt") == 2 for c in calls),
            "failed": [{"step": c["step"], "errors": c.get("errors", [])} for c in calls if c.get("errors")],
            "seconds": round(sum(c.get("seconds", 0) for c in calls), 2)}


def build(purpose, history, days, states, run_dir, stories=None):
    run_dir, stories = Path(run_dir), stories or {}
    tickets = {t.id: t for t in history.tickets}
    for n in sorted(states):
        tickets.update(states[n])
    result, prev_lessons = [], []
    for day in days:
        n, d = day.day, run_dir / f"day_{day.day:02}"
        state = states.get(n, {})
        plan = _load(d / "plan.json")
        feedback = {f["person"]: f for f in (_load(d / "feedback.json") or [])}
        score = _load(d / "score.json")
        lessons = _load(d / "lessons.json")
        plans, goal_hours = {}, {}
        for p in (plan or {}).get("plans", []):
            for item in p["top3"]:
                g = (state[item["ticket"]].goal if item["ticket"] in state else None) or "none"
                goal_hours[g] = goal_hours.get(g, 0) + float(item["hours"])
            f = feedback.get(p["person"])
            plans[p["person"]] = {
                "items": p["top3"], "lessons_used": p.get("lessons_used", []), "note": p.get("note", ""),
                "feedback": {"action": f["action"], "final": f["final"], "reason": f.get("reason")} if f else None,
                "score": (score or {}).get("people", {}).get(p["person"]),
            }
        result.append({
            "day": n, "date": day.date.isoformat(), "weekday": day.date.strftime("%A"),
            "label": day.date.strftime("%B %-d"), "story": stories.get(str(n), ""),
            "inputs": {"meetings": [m.model_dump(mode="json") for m in day.meetings],
                       "slack": [s.model_dump(mode="json") for s in day.slack],
                       "jira": [{**j.model_dump(mode="json", exclude={"set"}), "set": j.set.model_dump(mode="json", exclude_unset=True)}
                                for j in day.jira]},
            "sources": source_view(day),
            "brain": _load(d / "brain.json"),
            "planned": plan is not None,
            "plans": plans,
            "score": ({"mean": score["mean"], "accepts": score["accepts"], "responded": score.get("responded", len(history.people))}
                      if score and score.get("mean") is not None else None),
            "goal_hours": goal_hours,
            "open": [{"id": i, "owner": t.owner, "status": t.status} for i, t in
                     sorted(state.items(), key=lambda kv: int(kv[0].split("-")[1])) if t.status != "done"],
            "lessons_after": lessons,
            "lesson_changes": lesson_changes(prev_lessons, lessons, n) if lessons is not None else [],
            "health": (_load(d / "health.json") or {}).get("alerts"),
            "calls": calls_summary(run_dir / "llm_calls.jsonl", n),
        })
        if lessons is not None:
            prev_lessons = lessons
    return {
        "company": {"name": purpose.company, "mission": purpose.mission, "what_we_do": purpose.what_we_do,
                    "goals": [{"id": g.id, "rank": g.rank, "goal": g.goal, "metric": g.metric, "due": g.due.isoformat()}
                              for g in sorted(purpose.goals, key=lambda g: g.rank)]},
        "people": [{"id": p.id, "name": p.name, "first": p.name.split()[0], "role": p.role, "owns": p.owns, "notes": p.notes}
                   for p in history.people],
        "customers": [c.model_dump(mode="json") for c in history.customers],
        "oncall": [o.model_dump(mode="json") for o in history.oncall],
        "events": [e.model_dump(mode="json") for e in history.events],
        "tickets": {i: ticket_view(t) for i, t in tickets.items()},
        "lesson_health": _load(run_dir / "lesson_health.json") or {},
        "days": result,
    }
