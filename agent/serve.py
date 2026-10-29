"""Serve the supplied UI and replay one correction through Learn and Decide."""
import json, os, threading, time
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
from dotenv import load_dotenv
from agent import llm, steps, store
from agent.schemas import Brain, Feedback, LearnOutput, Lesson, Plan
from agent.workspace import Workspace, WorkspaceError
LIMIT_ERROR = "The live demo has reached today's limit. The Demo tab still works, or run it locally from GitHub."
KEY_ERROR = "Add your Anthropic API key to teach it something new. Replays of the demo answers work without one."
AUTH_ERROR = "That API key was rejected by the provider. Check it and try again."
IP_ERROR = "You have reached 10 new teach requests in the last hour. Try again later; demo replays still work."
ENV_KEY_ERROR = "No API key. Add ANTHROPIC_API_KEY to workflowgps/.env and restart, or paste a key in Settings."
WORKSPACE_OFF = "The Live workspace is turned off on this public server. Run it locally, or set ALLOW_WORKSPACE=1."
WORKSPACE_ROUTES = {"/api/workspace/setup": "setup", "/api/workspace/inputs": "add_input", "/api/workspace/inputs/remove": "remove_input",
                    "/api/workspace/plan": "plan", "/api/workspace/feedback": "feedback", "/api/workspace/close-day": "close_day",
                    "/api/workspace/reset": "reset", "/api/workspace/import": "import_"}
MODEL_ROUTES = {"plan", "close_day"}
def server(args):
    from agent.cli import run_path, validate
    purpose, history, days, states, _, facts = validate(args.data)
    llm.set_forbidden(facts)
    root, lock, ui = run_path(args.run), threading.Lock(), Path("ui").resolve()
    if not root.is_dir(): raise ValueError(f"Run does not exist: {root}")
    try:
        limit, port = int(os.getenv("TEACH_LIMIT", "40")), args.port if args.port is not None else int(os.getenv("PORT", "8000"))
        if limit < 0 or not 0 <= port <= 65535:
            raise ValueError
    except ValueError:
        raise ValueError("Invalid server port or daily limit") from None
    load_dotenv(Path(".env"))
    public = "PORT" in os.environ
    workspace_on = not public or os.getenv("ALLOW_WORKSPACE") == "1"
    workspace = Workspace(os.getenv("WORKSPACE_DIR", "workspaces/default"))
    usage, ips = [int(time.time() // 86400), 0, False], {}
    # Charge once at the first authenticated cache miss, never for replays.
    def limited(ip):
        now = time.time()
        if usage[0] != int(time.time() // 86400):
            usage[:2] = [int(time.time() // 86400), 0]
        if not usage[2]:
            for address in list(ips):
                ips[address] = [t for t in ips[address] if t > now - 3600]
                if not ips[address]: del ips[address]
            if len(ips.get(ip, [])) >= 10: raise BlockingIOError(IP_ERROR)
            if usage[1] >= limit:
                raise BlockingIOError(LIMIT_ERROR)
            ips.setdefault(ip, []).append(now)
            usage[1] += 1
            usage[2] = True
    def saved(day, name): return store.read_json(root / f"day_{day:02}" / f"{name}.json")
    def validate_request(data):
        if not isinstance(data, dict): raise ValueError("Expected a JSON object")
        day, person, final, reason = (data.get(k) for k in ("day", "person", "final", "reason"))
        if type(day) is not int or not 1 <= day < len(days): raise ValueError(f"day must be between 1 and {len(days) - 1}")
        if person not in [p.id for p in history.people]: raise ValueError("person must exist in history")
        if not isinstance(final, list) or len(final) != 3 or any(not isinstance(t, str) for t in final) or len(set(final)) != 3: raise ValueError("final must contain 3 distinct ticket IDs")
        if any(t not in states[day] or states[day][t].status == "done" for t in final): raise ValueError("final tickets must be open after today's Jira changes")
        plan = Plan.model_validate(saved(day, "plan"))
        proposed = [i.ticket for p in plan.plans if p.person == person for i in p.top3]
        if not isinstance(reason, str) or (final != proposed and not reason.strip()): raise ValueError("reason must be a string, nonempty for an edit")
        return day, person, final, reason, plan, proposed
    def teach(data, day, person, final, reason, plan, proposed):
        started = time.perf_counter()
        path = Path("runs/live") / datetime.now().strftime("%Y%m%d-%H%M%S")
        while path.exists():
            time.sleep(0.05)
            path = Path("runs/live") / datetime.now().strftime("%Y%m%d-%H%M%S")
        llm.start_run(path)
        store.write_json(path / "request.json", data)
        base = [Lesson.model_validate(l) for l in saved(day - 1, "lessons")] if day > 1 else []
        feedback = {f["person"]: Feedback.model_validate(f) for f in saved(day, "feedback")}
        feedback[person] = Feedback(person=person, action="accept" if final == proposed else "edit",
                                    proposed=proposed, final=final, reason=None if final == proposed else reason)
        edits = [feedback[p.id] for p in history.people if feedback[p.id].action == "edit"]
        output = steps.learn(history, days[day - 1], base, edits, states[day]) if edits else LearnOutput(ops=[])
        lessons = steps.apply_ops(base, output, day)
        sources = {key for d in days[:day + 1] for key, _ in store.source_items(d)}
        next_plan = steps.decide(purpose, history, days[day], Brain.model_validate(saved(day + 1, "brain")), states[day + 1], sources, lessons, plan)
        for name, value in (("ops", output.ops), ("lessons", lessons), ("plan", next_plan)):
            store.write_json(path / f"{name}.json", value)
        calls = [json.loads(line) for line in (path / "llm_calls.jsonl").read_text().splitlines()]
        return dict(day=day, next_day=day + 1, ops=output.ops, lessons=lessons, changed=[l.id for l in lessons if l not in base],
                    plan=next_plan, baseline=saved(day + 1, "plan"), calls=len(calls), cache_hits=sum(c["cache_hit"] for c in calls), seconds=round(time.perf_counter() - started, 2))
    def respond(handler, status, value):
        body = json.dumps(value, default=lambda v: v.model_dump(mode="json")).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        if handler.command != "HEAD":
            handler.wfile.write(body)
    def visitor(handler, teach=False):
        # Locally, a Live request without a key uses .env (None). Demo teach does so only when
        # TEACH_USE_ENV_KEY=1. The public server never spends its own key ("" means no key).
        key = handler.headers.get("X-LLM-Key", "").strip()
        if key or public or (teach and os.getenv("TEACH_USE_ENV_KEY") != "1"):
            return key
        return None
    def config():
        load_dotenv(Path(".env"))
        provider = os.environ.get("LLM_PROVIDER", "anthropic")
        env_key = bool(os.environ.get("ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"))
        return dict(ok=True, run=args.run, days=len(days), workspace=workspace_on, public=public,
                    env_key=env_key and not public, teach_env=env_key and not public and os.getenv("TEACH_USE_ENV_KEY") == "1", provider=provider, model=os.environ.get("LLM_MODEL"))
    def workspace_call(handler, name, data):
        key_token = llm.visitor_key.set(visitor(handler))
        try:
            with lock:
                result = getattr(workspace, name)(*([] if name in ("plan", "close_day", "reset") else [data]))
            return respond(handler, 200, result)
        except WorkspaceError as exc:
            return respond(handler, 400, dict(error=str(exc)))
        except (llm.MissingAPIKey, llm.RejectedAPIKey):
            return respond(handler, 401, dict(error=KEY_ERROR if public else AUTH_ERROR))
        except (llm.anthropic.AuthenticationError, llm.openai.AuthenticationError):
            return respond(handler, 401, dict(error=AUTH_ERROR))
        except RuntimeError as exc:
            message = str(exc)
            if message.startswith("Set ") and "API_KEY" in message:
                return respond(handler, 401, dict(error=ENV_KEY_ERROR))
            # Agent check failures name the step and the failed checks; never echo a key.
            return respond(handler, 502, dict(error=("The agent could not finish: " + message)[:600]))
        except Exception as exc:
            return respond(handler, 500, dict(error=f"Workspace request failed ({type(exc).__name__})."))
        finally:
            llm.visitor_key.reset(key_token)
    def get(handler):
        route = urlsplit(handler.path).path
        if route in ("/api/health", "/api/config"):
            return respond(handler, 200, config())
        if route.startswith("/api/workspace"):
            if not workspace_on:
                return respond(handler, 403, dict(error=WORKSPACE_OFF))
            try:
                with lock:
                    value = workspace.export() if route == "/api/workspace/export" else workspace.snapshot()
                return respond(handler, 200, value)
            except WorkspaceError as exc:
                return respond(handler, 400, dict(error=str(exc)))
        try:
            path = (ui / unquote(urlsplit(handler.path).path).lstrip("/")).resolve()
            path = (path / "index.html").resolve() if path.is_dir() else path
            if not path.is_relative_to(ui) or not path.is_file():
                raise ValueError
        except (OSError, ValueError):
            return respond(handler, 404, dict(error="Not found"))
        handler.translate_path = lambda _: str(path)
        getattr(SimpleHTTPRequestHandler, "do_" + handler.command)(handler)
    def post(handler):
        if handler.path in WORKSPACE_ROUTES:
            if not workspace_on:
                return respond(handler, 403, dict(error=WORKSPACE_OFF))
            try:
                size = int(handler.headers.get("Content-Length", "0"))
                if not 0 <= size <= 5_000_000: raise ValueError
                data = json.loads(handler.rfile.read(size) or b"{}")
            except ValueError:
                return respond(handler, 400, dict(error="Expected a JSON request of at most 5 MB"))
            return workspace_call(handler, WORKSPACE_ROUTES[handler.path], data)
        if handler.path != "/api/teach":
            return respond(handler, 404, dict(error="Unknown endpoint"))
        with lock:
            ip = handler.headers.get("X-Forwarded-For", "").split(",")[0].strip() or handler.client_address[0]
            key_token = llm.visitor_key.set(visitor(handler, teach=True))
            limit_token, usage[2] = llm.before_request.set(lambda: limited(ip)), False
            status = 400
            try:
                size = int(handler.headers.get("Content-Length", "0"))
                if not 0 < size <= 1_000_000: raise ValueError("Expected a JSON request of at most 1 MB")
                data = json.loads(handler.rfile.read(size))
                values = validate_request(data)
                status = 500
                result = teach(data, *values)
            except llm.MissingAPIKey:
                return respond(handler, 401, dict(error=KEY_ERROR))
            except llm.RejectedAPIKey:
                return respond(handler, 401, dict(error=AUTH_ERROR))
            except BlockingIOError as exc:
                return respond(handler, 429, dict(error=str(exc)))
            except RuntimeError as exc:
                if str(exc).startswith("Set ") and "API_KEY" in str(exc):
                    return respond(handler, 401, dict(error=ENV_KEY_ERROR))
                return respond(handler, status, dict(error="Invalid teach request" if status == 400 else "Teach failed; check the saved run for details."))
            except Exception:
                return respond(handler, status, dict(error="Invalid teach request" if status == 400 else "Teach failed; check the saved run for details."))
            finally:
                llm.before_request.reset(limit_token)
                llm.visitor_key.reset(key_token)
            respond(handler, 200, result)
    def handle(request, address, httpd):
        handler = SimpleHTTPRequestHandler.__new__(SimpleHTTPRequestHandler)
        handler.do_GET = handler.do_HEAD = lambda: get(handler)
        handler.do_POST, handler.log_message = lambda: post(handler), lambda *args: None
        handler.__init__(request, address, httpd, directory=str(ui))
    return ThreadingHTTPServer(("0.0.0.0" if "PORT" in os.environ else "127.0.0.1", port), handle)
