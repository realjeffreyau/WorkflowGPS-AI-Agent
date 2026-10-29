import json
from pathlib import Path
import pytest
from agent import llm, store
from agent.schemas import Brain, BrainItem, LearnOutput, Plan
from agent.workspace import Workspace, WorkspaceError

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
PEOPLE = ["Ada Lovelace", "Grace Hopper", "Alan Turing", "Katherine Johnson", "Linus Torvalds", "Margaret Hamilton"]


def setup_body(**extra):
    return {"company": "Acme", "mission": "Ship it", "what_we_do": "Widgets", "start_date": "2026-10-26",
            "goals": [{"goal": "Launch v2", "metric": "Live", "due": "2026-12-01"}],
            "people": [{"name": n, "role": "Engineer"} for n in PEOPLE],
            "customers": [{"name": "Big Co", "arr_share_pct": 40}],
            "tickets": [{"title": f"Task {i}", "owner": n.split()[0].lower(), "goal": "G1"} for i, n in enumerate(PEOPLE * 2)],
            **extra}


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "prompts").symlink_to(PROMPTS, target_is_directory=True)
    calls = []

    def canned(step, day, system, user, out, check):
        calls.append(step)
        if step == "interpret":
            source = next(s for s in (f"D{day}-S1", f"D{day}-M1", f"D{day}-J1") if s in user)
            result = Brain(summary="Today", items=[BrainItem(type="status", text="Moving", person=None, ticket=None, sources=[source])])
        elif step == "decide":
            _, history, _, states = Workspace(tmp_path / "w").load()
            open_ids = [t for t, v in states[day].items() if v.status != "done"]
            result = Plan.model_validate({"plans": [{"person": p.id, "lessons_used": [], "note": "Focus",
                "top3": [{"ticket": t, "hours": 2, "why": "Because", "sources": [t]} for t in open_ids[i:i + 3]]}
                for i, p in enumerate(history.people)]})
        else:
            result = LearnOutput(ops=[dict(op="add", rule="Do the review first", applies_to=["all"], from_people=["ada"])])
        assert check(result) == [], check(result)
        return result

    monkeypatch.setattr(llm, "call", canned)
    w = Workspace(tmp_path / "w")
    w.calls = calls
    return w


def test_setup_creates_valid_workspace(ws):
    snap = ws.setup(setup_body())
    assert snap["setup"] and len(snap["people"]) == 6 and snap["live"]["day"] == 1
    assert [p["id"] for p in snap["people"]] == ["ada", "grace", "alan", "katherine", "linus", "margaret"]
    assert len(snap["tickets"]) == 12 and "TW-1" in snap["tickets"]
    assert snap["days"][0]["date"] == "2026-10-26"
    assert store.validate_inputs(ws.root)[4] == {}


def test_setup_rejects_wrong_team_size(ws):
    body = setup_body()
    body["people"] = body["people"][:5]
    with pytest.raises(WorkspaceError, match="exactly six"):
        ws.setup(body)
    assert not ws.exists()


def test_invalid_input_rolls_back(ws):
    ws.setup(setup_body())
    before = (ws.root / "days/day_01.json").read_text()
    with pytest.raises(WorkspaceError):
        ws.add_input({"kind": "slack", "author": "nobody", "text": "hi"})
    assert (ws.root / "days/day_01.json").read_text() == before


def test_full_day_cycle_learns(ws):
    ws.setup(setup_body())
    with pytest.raises(WorkspaceError, match="Add at least one"):
        ws.plan()
    ws.add_input({"kind": "slack", "author": "ada", "channel": "#eng", "time": "09:10", "text": "Review is blocking Grace"})
    snap = ws.add_input({"kind": "jira", "action": "create", "by": "ada", "set": {"title": "Review PR", "owner": "ada", "priority": "high"}})
    assert "TW-13" in snap["tickets"]
    snap = ws.plan()
    assert snap["live"]["planned"] and not snap["live"]["stale"]
    proposed = [i["ticket"] for i in snap["days"][0]["plans"]["ada"]["items"]]
    ws.feedback({"person": "grace", "final": [i["ticket"] for i in snap["days"][0]["plans"]["grace"]["items"]]})
    with pytest.raises(WorkspaceError, match="Say why"):
        ws.feedback({"person": "ada", "final": proposed[::-1]})
    snap = ws.feedback({"person": "ada", "final": proposed[::-1], "reason": "Review first"})
    assert snap["live"]["responded"] == ["ada", "grace"]
    snap = ws.close_day()
    day1 = snap["days"][0]
    assert day1["score"]["responded"] == 2 and day1["score"]["accepts"] == 1
    assert [l["id"] for l in day1["lessons_after"]] == ["L1"] and day1["lesson_changes"][0]["from"] == ["ada"]
    assert snap["live"]["day"] == 2 and len(snap["days"]) == 2
    assert ws.calls == ["interpret", "decide", "learn"]
    with pytest.raises(WorkspaceError, match="Add at least one"):
        ws.plan()


def test_new_input_marks_plan_stale(ws):
    ws.setup(setup_body())
    ws.add_input({"kind": "meeting", "title": "Standup", "attendees": ["ada", "alan"], "transcript": "Ada: hi\nAlan: hello"})
    ws.plan()
    assert ws.add_input({"kind": "slack", "author": "alan", "text": "One more thing"})["live"]["stale"]


def test_export_import_roundtrip(ws, tmp_path):
    ws.setup(setup_body())
    ws.add_input({"kind": "slack", "author": "ada", "text": "hello"})
    bundle = json.loads(json.dumps(ws.export()))
    other = Workspace(tmp_path / "other")
    snap = other.import_(bundle)
    assert snap["company"]["name"] == "Acme" and snap["live"]["inputs"] == 1
    with pytest.raises(WorkspaceError, match="Unexpected file"):
        other.import_({"workflowgps_workspace": 1, "files": {**bundle["files"], "../evil.json": "{}"}})


def test_demo_export_never_contains_hidden_facts():
    root = Path(__file__).resolve().parent.parent
    text = (root / "ui/demo.json").read_text()
    facts = [r["hidden_fact"] for p in (root / "data/eval").glob("*.json") for r in json.loads(p.read_text())["records"]]
    assert facts and not any(f in text for f in facts if len(f) >= 20)
