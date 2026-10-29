"""Live workspace: one real team's setup, daily inputs, plans, feedback and learning, kept on disk.

Layout matches data/ plus a run folder, so the agent's own validation, steps and view are reused:
    workspaces/default/purpose.json, history.json, days/day_NN.json
    workspaces/default/runs/day_NN/{brain,plan,feedback,score,lessons,health}.json
    workspaces/default/runs/llm_calls.jsonl, lesson_health.json
"""
import hashlib
import json
import re
import shutil
from datetime import date, timedelta
from pathlib import Path
from agent import llm, steps, store, view
from agent.evaluate import score_person
from agent.schemas import Brain, Feedback, Lesson, LearnOutput, Plan

SLACK_CHANNELS = ("#general", "#eng", "#design", "#customers")
TICKET_FIELDS = ("title", "description", "owner", "status", "priority", "goal", "customer", "due", "estimate_hours")


class WorkspaceError(ValueError):
    """A request the workspace cannot accept; the message is safe to show."""


def _clean(value):
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


def _slug(text, taken, pattern="[^a-z]", sep=""):
    base = re.sub(pattern, sep, text.lower()).strip("-") or "x"
    slug = base
    while slug in taken:
        slug += "-x" if sep else "x"
    taken.add(slug)
    return slug


def _flatten(errors):
    return "; ".join(f"{Path(f).name}: {m}" for f, ms in errors.items() for m in ms[:3])[:1200]


class Workspace:
    def __init__(self, root):
        self.root = Path(root)
        self.runs = self.root / "runs"

    # ---------- files ----------
    def exists(self):
        return (self.root / "purpose.json").exists() and (self.root / "history.json").exists()

    def _day_path(self, n):
        return self.root / "days" / f"day_{n:02}.json"

    def _run(self, n):
        return self.runs / f"day_{n:02}"

    def _read(self, path, default=None):
        path = Path(path)
        return json.loads(path.read_text()) if path.exists() else default

    def load(self):
        if not self.exists():
            raise WorkspaceError("Set up your team first.")
        purpose, history, days, states, errors = store.validate_inputs(self.root)
        if errors:
            raise WorkspaceError(_flatten(errors))
        return purpose, history, days, states

    def _snapshot_files(self):
        files = [self.root / "purpose.json", self.root / "history.json", *sorted((self.root / "days").glob("*.json"))]
        return {p: p.read_text() for p in files if p.exists()}

    def _commit(self, mutate):
        """Apply mutate(), then validate the whole workspace; roll back on any error."""
        before = self._snapshot_files()
        try:
            mutate()
            _, _, _, _, errors = store.validate_inputs(self.root)
            if errors:
                raise WorkspaceError(_flatten(errors))
        except Exception:
            for p in self._snapshot_files():
                if p not in before:
                    p.unlink()
            for p, text in before.items():
                p.write_text(text)
            raise

    def _current(self, days):
        n = days[-1].day
        if (self._run(n) / "lessons.json").exists():
            raise WorkspaceError(f"Day {n} is already closed.")
        return days[-1]

    @staticmethod
    def _inputs_hash(day):
        return hashlib.sha256(day.model_dump_json(exclude={"day", "date"}).encode()).hexdigest()

    # ---------- view ----------
    def snapshot(self):
        if not self.exists():
            return {"setup": False}
        purpose, history, days, states = self.load()
        data = view.build(purpose, history, days, states, self.runs)
        day = days[-1]
        run = self._run(day.day)
        planned = (run / "plan.json").exists()
        feedback = self._read(run / "feedback.json", [])
        data["setup"] = True
        data["live"] = {
            "day": day.day,
            "planned": planned,
            "stale": planned and self._read(run / "inputs.json", {}).get("hash") != self._inputs_hash(day),
            "responded": [f["person"] for f in feedback],
            "inputs": sum(len(x) for x in (day.meetings, day.slack, day.jira)),
            "open_tickets": sum(t.status != "done" for t in states[day.day].values()),
        }
        data["setup_raw"] = {"purpose": self._read(self.root / "purpose.json"), "history": self._read(self.root / "history.json"),
                             "start_date": days[0].date.isoformat()}
        return data

    # ---------- setup ----------
    def setup(self, body):
        if not isinstance(body, dict):
            raise WorkspaceError("Expected a JSON object")
        company, mission = _clean(body.get("company")), _clean(body.get("mission"))
        if not company or not mission:
            raise WorkspaceError("Company name and mission are required.")
        goals = [g for g in body.get("goals") or [] if _clean(g.get("goal"))]
        if not goals:
            raise WorkspaceError("Add at least one team goal.")
        people_in = body.get("people") or []
        if len(people_in) != 6 or any(not _clean(p.get("name")) for p in people_in):
            raise WorkspaceError("WorkflowGPS plans for exactly six people. Give all six a name.")
        existing = self._read(self.root / "history.json")
        try:
            start = date.fromisoformat(body.get("start_date") or date.today().isoformat())
        except ValueError:
            raise WorkspaceError("Start date must be YYYY-MM-DD") from None

        taken, people = set(), []
        for p in people_in:
            pid = _clean(p.get("id"))
            if pid and re.fullmatch(r"[a-z]+", pid) and pid not in taken:
                taken.add(pid)
            else:
                pid = _slug(p["name"].split()[0], taken)
            people.append({"id": pid, "name": _clean(p["name"]), "role": _clean(p.get("role")) or "Teammate",
                           "owns": _clean(p.get("owns")) or "", "notes": _clean(p.get("notes")) or ""})
        ids = {p["id"] for p in people}
        ctaken, customers = set(), []
        for c in body.get("customers") or []:
            if not _clean(c.get("name")):
                continue
            cid = _clean(c.get("id"))
            if cid and cid not in ctaken:
                ctaken.add(cid)
            else:
                cid = _slug(c["name"], ctaken, "[^a-z0-9]+", "-")
            customers.append({"id": cid, "name": _clean(c["name"]), "arr_share_pct": float(c.get("arr_share_pct") or 0),
                              "renewal_date": _clean(c.get("renewal_date")) or (start + timedelta(days=365)).isoformat(),
                              "health": c.get("health") if c.get("health") in ("green", "yellow", "red") else "green",
                              "contact": _clean(c.get("contact")) or "", "notes": _clean(c.get("notes")) or ""})
        oncall = [{"week_of": o["week_of"], "person": o["person"]} for o in body.get("oncall") or []
                  if _clean(o.get("week_of")) and o.get("person") in ids]
        tickets, n = [], 0
        used = {t.get("id") for t in body.get("tickets") or [] if _clean(t.get("id"))}
        for t in body.get("tickets") or []:
            if not _clean(t.get("title")):
                continue
            tid = _clean(t.get("id"))
            if not tid:
                n += 1
                while f"TW-{n}" in used:
                    n += 1
                tid = f"TW-{n}"
                used.add(tid)
            est = t.get("estimate_hours")
            tickets.append({"id": tid, "title": _clean(t["title"]), "description": _clean(t.get("description")) or "",
                            "owner": t.get("owner") if t.get("owner") in ids else people[0]["id"],
                            "status": t.get("status") if t.get("status") in ("todo", "in_progress", "blocked", "done") else "todo",
                            "priority": t.get("priority") if t.get("priority") in ("high", "medium", "low") else "medium",
                            "goal": _clean(t.get("goal")), "customer": _clean(t.get("customer")), "due": _clean(t.get("due")),
                            "estimate_hours": float(est) if est not in (None, "") else None})
        events = [{"date": e["date"], "text": _clean(e["text"])} for e in body.get("events") or []
                  if _clean(e.get("date")) and _clean(e.get("text"))]
        purpose = {"company": company, "what_we_do": _clean(body.get("what_we_do")) or company, "mission": mission,
                   "goals": [{"id": f"G{i}", "rank": i, "goal": _clean(g["goal"]), "metric": _clean(g.get("metric")) or "Done",
                              "due": _clean(g.get("due")) or (start + timedelta(days=30)).isoformat()} for i, g in enumerate(goals, 1)]}
        as_of = existing["as_of"] if existing and self._day_path(1).exists() else (start - timedelta(days=1)).isoformat()
        history = {"as_of": as_of, "people": people, "customers": customers, "oncall": oncall, "tickets": tickets, "events": events}

        def mutate():
            self.root.mkdir(parents=True, exist_ok=True)
            store.write_json(self.root / "purpose.json", purpose)
            store.write_json(self.root / "history.json", history)
            if not self._day_path(1).exists():
                store.write_json(self._day_path(1), {"day": 1, "date": start.isoformat(), "meetings": [], "slack": [], "jira": []})
        self._commit(mutate)
        return self.snapshot()

    # ---------- inputs ----------
    def add_input(self, body):
        _, history, days, states = self.load()
        day = self._current(days)
        kind = body.get("kind") if isinstance(body, dict) else None
        raw = self._read(self._day_path(day.day))
        ids = {p.id for p in history.people}
        if kind == "meeting":
            transcript = body.get("transcript") or []
            if isinstance(transcript, str):
                transcript = [line.strip() for line in transcript.splitlines() if line.strip()]
            if not _clean(body.get("title")) or not transcript:
                raise WorkspaceError("A meeting needs a title and some notes or transcript.")
            raw["meetings"].append({"title": _clean(body["title"]), "time": _clean(body.get("time")) or "09:00",
                                    "attendees": [a for a in body.get("attendees") or [] if a in ids] or [history.people[0].id],
                                    "transcript": transcript})
        elif kind == "slack":
            if not _clean(body.get("text")):
                raise WorkspaceError("A message needs text.")
            raw["slack"].append({"channel": body.get("channel") if body.get("channel") in SLACK_CHANNELS else "#general",
                                 "author": body.get("author"), "time": _clean(body.get("time")) or "09:00", "text": _clean(body["text"])})
        elif kind == "jira":
            action = body.get("action")
            patch = {k: _clean(v) for k, v in (body.get("set") or {}).items() if k in TICKET_FIELDS and _clean(v) is not None}
            try:
                if "estimate_hours" in patch:
                    patch["estimate_hours"] = float(patch["estimate_hours"])
            except (TypeError, ValueError):
                raise WorkspaceError("Estimate must be a number of hours.") from None
            if action == "create":
                n = max([int(k.split("-")[1]) for k in states[day.day]] or [0]) + 1
                ticket = f"TW-{n}"
                patch.setdefault("status", "todo")
                patch.setdefault("priority", "medium")
            elif action == "update":
                ticket = body.get("ticket")
            else:
                raise WorkspaceError("Ticket change must be create or update.")
            raw["jira"].append({"ticket": ticket, "action": action, "by": body.get("by"), "set": patch,
                                "comment": _clean(body.get("comment"))})
        else:
            raise WorkspaceError("Input kind must be meeting, slack or jira.")
        self._commit(lambda: store.write_json(self._day_path(day.day), raw))
        return self.snapshot()

    def remove_input(self, body):
        _, _, days, _ = self.load()
        day = self._current(days)
        raw = self._read(self._day_path(day.day))
        key = {"meeting": "meetings", "slack": "slack", "jira": "jira"}.get(body.get("kind"))
        index = body.get("index")
        if not key or type(index) is not int or not 0 <= index < len(raw[key]):
            raise WorkspaceError("Unknown input.")
        raw[key].pop(index)
        self._commit(lambda: store.write_json(self._day_path(day.day), raw))
        return self.snapshot()

    # ---------- agent steps ----------
    def _llm(self):
        self.runs.mkdir(parents=True, exist_ok=True)
        (self.runs / "llm_calls.jsonl").touch()
        llm._run_dir = self.runs

    def _call(self, fn, *args):
        """Run an agent step against this workspace's call log. Hidden demo facts don't apply to a real team."""
        self._llm()
        forbidden, llm._forbidden = llm._forbidden, []
        try:
            return fn(*args)
        finally:
            llm._forbidden = forbidden

    def plan(self):
        purpose, history, days, states = self.load()
        day = self._current(days)
        n = day.day
        if not (day.meetings or day.slack or day.jira):
            raise WorkspaceError("Add at least one meeting, message or ticket change for today first.")
        tickets = states[n]
        if sum(t.status != "done" for t in tickets.values()) < 3:
            raise WorkspaceError("Each person needs three open tickets to choose from. Add more tickets first.")
        sources = {key for d in days for key, _ in store.source_items(d)}
        prev = self._run(n - 1)
        brain = Brain.model_validate(self._read(prev / "brain.json")) if n > 1 and (prev / "brain.json").exists() else None
        lessons = [Lesson.model_validate(l) for l in self._read(prev / "lessons.json", [])] if n > 1 else []
        previous = Plan.model_validate(self._read(prev / "plan.json")) if n > 1 and (prev / "plan.json").exists() else None
        brain = self._call(steps.interpret, purpose, history, day, brain, tickets, sources)
        plan = self._call(steps.decide, purpose, history, day, brain, tickets, sources, lessons, previous)
        run = self._run(n)
        store.write_json(run / "brain.json", brain)
        store.write_json(run / "plan.json", plan)
        store.write_json(run / "inputs.json", {"hash": self._inputs_hash(day)})
        (run / "feedback.json").unlink(missing_ok=True)
        return self.snapshot()

    def feedback(self, body):
        _, history, days, states = self.load()
        day = self._current(days)
        run = self._run(day.day)
        if not (run / "plan.json").exists():
            raise WorkspaceError("Generate today's plans first.")
        person = body.get("person") if isinstance(body, dict) else None
        if person not in {p.id for p in history.people}:
            raise WorkspaceError("Unknown person.")
        order = [p.id for p in history.people]
        current = [f for f in self._read(run / "feedback.json", []) if f["person"] != person]
        if not body.get("clear"):
            plan = Plan.model_validate(self._read(run / "plan.json"))
            proposed = [i.ticket for p in plan.plans if p.person == person for i in p.top3]
            final, reason = body.get("final"), _clean(body.get("reason"))
            if not isinstance(final, list) or len(final) != 3 or len(set(final)) != 3:
                raise WorkspaceError("Pick exactly three different tickets.")
            tickets = states[day.day]
            if any(not isinstance(t, str) or t not in tickets or tickets[t].status == "done" for t in final):
                raise WorkspaceError("Every stop must be an open ticket.")
            edited = final != proposed
            if edited and not reason:
                raise WorkspaceError("Say why in one sentence, so the agent can learn from it.")
            current.append(Feedback(person=person, action="edit" if edited else "accept", proposed=proposed, final=final,
                                    reason=reason if edited else None).model_dump(mode="json"))
        store.write_json(run / "feedback.json", sorted(current, key=lambda f: order.index(f["person"])))
        return self.snapshot()

    def close_day(self):
        _, history, days, states = self.load()
        day = self._current(days)
        n, run = day.day, self._run(day.day)
        if not (run / "plan.json").exists():
            raise WorkspaceError("Generate today's plans before closing the day.")
        plan = Plan.model_validate(self._read(run / "plan.json"))
        feedback = [Feedback.model_validate(f) for f in self._read(run / "feedback.json", [])]
        lessons = [Lesson.model_validate(l) for l in self._read(self._run(n - 1) / "lessons.json", [])] if n > 1 else []
        edits = [f for f in feedback if f.action == "edit"]
        output = self._call(steps.learn, history, day, lessons, edits, states[n]) if edits else LearnOutput(ops=[])
        new_lessons = steps.apply_ops(lessons, output, n)
        people = {f.person: score_person(f.proposed, f.final) for f in feedback}
        score = {"day": n, "date": day.date.isoformat(), "mean": round(sum(people.values()) / len(people), 1) if people else None,
                 "accepts": sum(f.action == "accept" for f in feedback), "responded": len(feedback), "people": people}
        store.write_json(run / "score.json", score)
        store.write_json(run / "lessons.json", new_lessons)
        self._health(n, plan, feedback, score, new_lessons)
        next_date = max(day.date + timedelta(days=1), date.today())
        store.write_json(self._day_path(n + 1), {"day": n + 1, "date": next_date.isoformat(), "meetings": [], "slack": [], "jira": []})
        return self.snapshot()

    def _health(self, n, plan, feedback, score, lessons):
        alerts, edits = [], {f.person for f in feedback if f.action == "edit"}
        def alert(kind, detail, person=None, lesson=None):
            alerts.append(dict(kind=kind, person=person, lesson=lesson, detail=detail))
        log = self.runs / "llm_calls.jsonl"
        calls = [json.loads(l) for l in log.read_text().splitlines() if l.strip()] if log.exists() else []
        failures = {c["step"]: c["errors"] for c in calls if c["day"] == n and c["attempt"] == 1 and c.get("errors")}
        for c in calls:
            if c["day"] == n and c["attempt"] == 2 and not c.get("errors") and c["step"] in failures:
                alert("repaired", f"{c['step']}: {'; '.join(failures[c['step']])}")
        for person, value in score["people"].items():
            if value < 50:
                alert("rejected_plan", f"{person} scored {value:.1f}", person)
        before = self._read(self._run(n - 1) / "score.json") if n > 1 else None
        if before and before.get("mean") is not None and score["mean"] is not None and before["mean"] - score["mean"] >= 15:
            alert("regression", f"mean {score['mean']:.1f}, down {before['mean'] - score['mean']:.1f} from day {n - 1}")
        totals = self._read(self.runs / "lesson_health.json", {})
        for p in plan.plans:
            for lesson in dict.fromkeys(p.lessons_used):
                total = totals.setdefault(lesson, dict(uses=0, overrides=0, flagged=False))
                total["uses"] += 1
                total["overrides"] += p.person in edits
                if p.person in edits:
                    alert("lesson_overridden", f"{p.person} edited despite using {lesson}", p.person, lesson)
        for lesson in lessons:
            totals.setdefault(lesson.id, dict(uses=0, overrides=0, flagged=False))
        for total in totals.values():
            total["flagged"] = total["overrides"] >= 2 and total["overrides"] * 2 >= total["uses"]
        store.write_json(self._run(n) / "health.json", dict(day=n, alerts=alerts))
        store.write_json(self.runs / "lesson_health.json", totals)

    # ---------- admin ----------
    def reset(self):
        if self.root.exists():
            shutil.rmtree(self.root)
        return {"setup": False}

    def export(self):
        if not self.exists():
            raise WorkspaceError("Nothing to export yet.")
        files = {p.relative_to(self.root).as_posix(): p.read_text() for p in sorted(self.root.rglob("*"))
                 if p.is_file() and p.suffix in (".json", ".jsonl")}
        return {"workflowgps_workspace": 1, "files": files}

    def import_(self, body):
        files = body.get("files") if isinstance(body, dict) and body.get("workflowgps_workspace") == 1 else None
        if not isinstance(files, dict) or "purpose.json" not in files or "history.json" not in files:
            raise WorkspaceError("Not a WorkflowGPS workspace export.")
        for name, text in files.items():
            if not isinstance(text, str) or not re.fullmatch(
                    r"(purpose|history)\.json|days/day_\d{2,}\.json|runs/(day_\d{2,}/)?[a-z_]+\.jsonl?", name):
                raise WorkspaceError(f"Unexpected file in export: {name}")
        staging = self.root.with_name(self.root.name + ".import")
        if staging.exists():
            shutil.rmtree(staging)
        for name, text in files.items():
            (staging / name).parent.mkdir(parents=True, exist_ok=True)
            (staging / name).write_text(text)
        _, _, _, _, errors = store.validate_inputs(staging)
        if errors:
            shutil.rmtree(staging)
            raise WorkspaceError(_flatten(errors))
        if self.root.exists():
            shutil.rmtree(self.root)
        staging.rename(self.root)
        return self.snapshot()
