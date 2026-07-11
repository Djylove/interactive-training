"""Data plane: session events -> a local Aim repo (frontend plan §4).

Rank-0 only, one ``aim.Run`` per round, consumed on a daemon thread so tracking
never blocks the training loop. The run's ``control`` param carries the trainer's
HTTP control URL, which is how the Aim frontend finds the live process.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import subprocess
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)

_TEXT_EVENTS = frozenset({
    "knob_changed", "action_result", "evaluate_requested", "checkpoint_saved",
    "checkpoint_loaded", "module_reset", "note", "status_changed",
    "agent_plan", "agent_reflection", "agent_attached", "agent_enabled",
    "agent_configured", "context_changed", "agent_call",
})

_NON_METRIC_KEYS = frozenset({"step", "eval"})


def _is_rank0() -> bool:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    for var in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        val = os.environ.get(var)
        if val is not None:
            try:
                return int(val) == 0
            except ValueError:
                pass
    return True


class AimTransport:
    def __init__(self, repo: str | None = None, experiment: str | None = None,
                 control_url: str | Callable[[], str] | None = None):
        self.repo = repo
        self.experiment = experiment
        self.control_url = control_url
        self._session: Any = None
        self._queue = None
        self._thread: threading.Thread | None = None
        self._run = None
        self._round = 0

    def start(self, session: Any) -> None:
        if not _is_rank0():
            return
        self._session = session
        # Open the round-0 run up front so the repo exists and the control URL is
        # published before training starts — the Aim UI is usable immediately.
        self._ensure_run()
        self._queue = session.events.subscribe(since=0)
        self._thread = threading.Thread(target=self._pump, name="aim-transport", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._queue is None:
            return
        self._session.events.unsubscribe(self._queue)
        self._queue.put(None)
        self._thread.join(timeout=10)
        self._close_run()
        self._queue = None

    def _pump(self) -> None:
        while True:
            ev = self._queue.get()
            if ev is None:
                break
            try:
                self._handle(ev)
            except Exception:
                logger.exception("AimTransport: failed to handle %r event", ev.type)

    def _handle(self, ev) -> None:
        if ev.type == "metrics":
            self._track_metrics(ev)
        elif ev.type == "round_started":
            self._begin_round(int(ev.payload.get("round", self._round)))
        elif ev.type == "round_finished":
            score = ev.payload.get("score")
            if score is not None:
                self._ensure_run().track(float(score), name="round_score",
                                         step=int(ev.payload.get("round", self._round)))
        elif ev.type == "model_tree":
            # Persist the tree as a run param so the frontend's archive mode can render
            # the Model panel after the process is gone (design §6.7 persistence gap).
            tree = ev.payload.get("tree")
            if tree is not None:
                self._ensure_run()["model_tree"] = tree
        if ev.type in _TEXT_EVENTS:
            self._track_text(ev)
        if ev.type == "status_changed":
            self._ensure_run()["status"] = ev.payload.get("status")

    def _track_metrics(self, ev) -> None:
        run = self._ensure_run()
        context = {"round": self._round, "branch": ev.branch_id}
        if ev.payload.get("eval"):
            context["subset"] = "eval"
        for name, value in ev.payload.items():
            if name in _NON_METRIC_KEYS or isinstance(value, bool) \
                    or not isinstance(value, (int, float)):
                continue
            run.track(value, name=name, step=ev.payload.get("step"), context=context)

    def _track_text(self, ev) -> None:
        from aim import Text
        body = json.dumps({"type": ev.type, **ev.payload}, default=str)
        self._ensure_run().track(Text(body), name="agent_actions", step=ev.seq,
                                 context={"type": ev.type, "branch": ev.branch_id})

    def _begin_round(self, round_idx: int) -> None:
        if self._run is not None and round_idx == self._round:
            return
        self._close_run()
        self._round = round_idx
        self._ensure_run()

    def _ensure_run(self):
        if self._run is None:
            self._run = self._open_run()
        return self._run

    def _open_run(self):
        run = self._create_run()
        run["round"] = self._round
        session = self._session
        if session is not None:
            # Round budget as run params so archive mode can render the rail/progress
            # strip without a live /state (multiround_ux §3.3). Stable across the run.
            baseline_rounds = getattr(session, "_baseline_rounds", 0) or 0
            total = baseline_rounds + (getattr(session, "max_rounds", 0) or 0)
            if total:
                run["rounds_total"] = total
            run["baseline"] = self._round < baseline_rounds
        if self._session is not None and self._session.goal is not None:
            run["goal"] = self._session.goal.model_dump()
        url = self.control_url() if callable(self.control_url) else self.control_url
        if url:
            run["control"] = {"url": url}
        logger.info("AimTransport: round %d -> run %s (repo=%s)",
                    self._round, getattr(run, "hash", "?"), self.repo or "<default>")
        return run

    def _create_run(self):
        from aim import Repo, Run
        if self.repo and not Repo.exists(self.repo):
            Repo.from_path(self.repo, init=True)
        return Run(repo=self.repo,
                   experiment=self.experiment or getattr(self._session, "run_id", None))

    def _close_run(self) -> None:
        if self._run is None:
            return
        try:
            self._run.close()
        except Exception:
            logger.exception("AimTransport: failed to close run")
        self._run = None


class AimUp:
    """Serves the Aim web UI (`aim up`) as a child process for the session's lifetime,
    so the frontend is browsable immediately — no manual `aim up` step."""

    def __init__(self, repo: str | None = None, host: str = "0.0.0.0", port: int = 43800):
        self.repo = repo
        self.host = host
        self.port = port
        self._proc: subprocess.Popen | None = None
        self._ready = False
        self._log_thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host == "0.0.0.0" else self.host
        return f"http://{host}:{self.port}/live"

    @property
    def _health_url(self) -> str:
        host = "127.0.0.1" if self.host == "0.0.0.0" else self.host
        return f"http://{host}:{self.port}/"

    def start(self, session: Any) -> None:
        if not _is_rank0():
            return
        if self.repo:
            try:
                from aim import Repo
                if not Repo.exists(self.repo):
                    Repo.from_path(self.repo, init=True)
            except Exception:
                logger.exception("AimUp: could not initialize aim repo %s", self.repo)
        cmd = ["aim", "up", "--host", self.host, "--port", str(self.port)]
        if self.repo:
            cmd += ["--repo", self.repo]
        try:
            # Pipe `aim up`'s own output so its startup log is visible instead of
            # being swallowed; a daemon reader forwards each line to the logger.
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT, text=True,
                                          bufsize=1, start_new_session=True)
        except OSError:
            logger.exception("AimUp: could not launch `aim up` (is aim installed?)")
            return
        atexit.register(self.stop)  # safety net if session.end() never runs
        self._log_thread = threading.Thread(target=self._stream_logs,
                                            name="aim-up-log", daemon=True)
        self._log_thread.start()
        # Block until the UI actually answers HTTP so callers only advertise it
        # (and the run only proceeds) once it is truly serving.
        from interactive_training.transport.composite import wait_http_ready
        if not wait_http_ready(self._health_url, timeout=60):
            logger.warning("AimUp: web UI not responding at %s (pid=%d)",
                           self.url, self._proc.pid)
            return
        self._ready = True
        logger.info("AimUp: web UI ready at %s (repo=%s, pid=%d)",
                    self.url, self.repo or "<default>", self._proc.pid)

    def _stream_logs(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    logger.info("[aim up] %s", line)
        except Exception:
            pass

    def stop(self) -> None:
        if self._proc is None:
            return
        pgid = self._proc.pid  # == process-group id thanks to start_new_session
        self._signal_group(pgid, signal.SIGTERM)
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        self._signal_group(pgid, signal.SIGKILL)
        self._proc = None

    @staticmethod
    def _signal_group(pgid: int, sig: int) -> None:
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass  # whole group already gone
        except OSError:
            logger.exception("AimUp: failed to signal process group %d", pgid)
