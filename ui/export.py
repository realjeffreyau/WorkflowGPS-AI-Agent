"""Export run results for the UI's Demo mode.

Reads data/ and the finished runs, and writes ui/demo.js (window.DEMO_DATA) and ui/demo.json.
Never reads data/eval/, so no hidden fact can reach the page.

    python ui/export.py                       # runs/learn and runs/nolearn
    python ui/export.py --learn learn_s1 --nolearn nolearn_s1
"""
import argparse
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agent import store, view  # noqa: E402


def scores(run_dir):
    path = run_dir / "scores.csv"
    if not path.exists():
        return None
    return {int(r["day"]): float(r["mean"]) for r in csv.DictReader(path.read_text().splitlines())}


def item_acceptance(feedback):
    shares = [len(set(f["proposed"]) & set(f["final"])) / len(f["proposed"]) for f in feedback if f["proposed"]]
    return round(100 * sum(shares) / len(shares), 1) if shares else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--learn", default="learn")
    ap.add_argument("--nolearn", default="nolearn")
    ap.add_argument("--out", default=str(ROOT / "ui"))
    a = ap.parse_args()

    data, runs, out = pathlib.Path(a.data), pathlib.Path(a.runs), pathlib.Path(a.out)
    purpose, history, days, states, errors = store.validate_inputs(data)
    if errors:
        raise SystemExit(f"data/ is invalid: {errors}")
    story_path = out / "story.json"
    stories = json.loads(story_path.read_text()) if story_path.exists() else {}
    learn, nolearn = runs / a.learn, runs / a.nolearn
    days = [d for d in days if (learn / f"day_{d.day:02}" / "plan.json").exists()]
    demo = view.build(purpose, history, days, states, learn, stories)

    for d in demo["days"]:
        n = d["day"]
        nd = nolearn / f"day_{n:02}"
        ns = json.loads((nd / "score.json").read_text()) if (nd / "score.json").exists() else None
        d["nolearn"] = {"mean": ns["mean"], "accepts": ns["accepts"]} if ns else None
        fb = learn / f"day_{n:02}" / "feedback.json"
        if d["score"] and fb.exists():
            d["score"]["item_acceptance"] = item_acceptance(json.loads(fb.read_text()))

    series = {}
    for name, label in ((a.learn, "With learning"), (a.nolearn, "Without learning"),
                        ("ranker_online", "Weight ranker"), ("ranker_fixed", "Fixed ranker")):
        s = scores(runs / name)
        if s:
            series[name] = {"label": label, "by_day": s}
    bands = {}
    for mode in ("learn", "nolearn"):
        seeds = [s for s in (scores(runs / f"{mode}_s{i}") for i in (1, 2, 3)) if s]
        if seeds:
            bands[mode] = {str(n): [min(s[n] for s in seeds if n in s), max(s[n] for s in seeds if n in s)] for n in seeds[0]}
    demo["compare"] = {"learn": a.learn, "nolearn": a.nolearn, "series": series, "bands": bands}

    text = json.dumps(demo, indent=1, ensure_ascii=False)
    (out / "demo.json").write_text(text + "\n")
    (out / "demo.js").write_text("window.DEMO_DATA = " + text + ";\n")
    print(f"wrote {out / 'demo.js'}: {len(demo['days'])} days, runs {a.learn} and {a.nolearn}")


if __name__ == "__main__":
    main()
