"""Training state, branch tree, checkpoint registry (plan §3.5, §6)."""
from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


class Checkpoint(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    path: str
    step: int
    tag: str | None = None
    branch_id: str = "main"
    ts: float = Field(default_factory=time.time)


class CheckpointRegistry:
    """Records checkpoints from the path the loop actually saved to (fixes P0.6)."""

    def __init__(self):
        self._by_id: dict[str, Checkpoint] = {}
        self._order: list[str] = []

    def add(self, path: str, step: int, branch_id: str = "main", tag: str | None = None) -> Checkpoint:
        ckpt = Checkpoint(path=path, step=step, branch_id=branch_id, tag=tag)
        self._by_id[ckpt.id] = ckpt
        self._order.append(ckpt.id)
        return ckpt

    def get(self, ckpt_id: str) -> Checkpoint | None:
        return self._by_id.get(ckpt_id)

    def list(self) -> list[Checkpoint]:
        return [self._by_id[i] for i in self._order]


class Branch(BaseModel):
    id: str
    parent: str | None = None
    from_checkpoint: str | None = None
    created_at: float = Field(default_factory=time.time)


class BranchTree:
    """Pure bookkeeping; fork never returns None (fixes P0.7, decouples P1.8)."""

    def __init__(self, root: str = "main"):
        self.root = root
        self._branches: dict[str, Branch] = {root: Branch(id=root)}
        self.current = root

    def fork(self, parent: str, from_checkpoint: str | None = None) -> Branch:
        new_id = f"{parent}/{uuid.uuid4().hex[:8]}"
        branch = Branch(id=new_id, parent=parent, from_checkpoint=from_checkpoint)
        self._branches[new_id] = branch
        self.current = new_id
        return branch

    def list(self) -> list[Branch]:
        return list(self._branches.values())


class TrainingState:
    def __init__(self):
        self.status: str = "idle"
        self.history: list[dict] = []
        self.step: int = 0
        self.branches = BranchTree()
        self.checkpoints = CheckpointRegistry()
        self.model_tree: dict | None = None

    @property
    def branch_id(self) -> str:
        return self.branches.current

    def log(self, metrics: dict, step: int | None = None) -> dict:
        record = dict(metrics)
        if step is not None:
            self.step = step
        record.setdefault("step", self.step)
        self.history.append(record)
        return record

    def recent(self, n: int = 10) -> list[dict]:
        return self.history[-n:]

    def snapshot(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "step": self.step,
            "branch_id": self.branch_id,
            "branches": [b.model_dump() for b in self.branches.list()],
            "checkpoints": [c.model_dump() for c in self.checkpoints.list()],
            "model_tree": self.model_tree,
        }


def build_model_tree(model: Any) -> dict:
    """BFS over `model.named_modules()` into a name/children/module_type tree
    (reusable helper; replaces the old in-callback `_parse_model_tree`, P2.6)."""
    from collections import deque

    names, types = [], {}
    for name, module in model.named_modules():
        if name.strip() == "":
            continue
        names.append(name)
        types[name] = module.__class__.__name__

    children: dict[str, list[str]] = {}
    parent: dict[str, str] = {}
    for parts in sorted((n.split(".") for n in names), key=len):
        name = ".".join(parts)
        children.setdefault(name, [])
        if len(parts) > 1:
            par = ".".join(parts[:-1])
            children.setdefault(par, []).append(name)
            parent[name] = par

    roots = [n for n in names if n not in parent]
    root = {"name": "Model", "module_type": type(model).__name__, "children": []}
    if len(roots) == 1:
        root["name"] = roots[0]
        root["module_type"] = types.get(roots[0], "Unknown")
    elif roots:
        children["Model"] = roots

    q = deque([root])
    while q:
        node = q.popleft()
        for child_name in children.get(node["name"], []):
            child = {"name": child_name, "module_type": types.get(child_name, "Unknown"), "children": []}
            node["children"].append(child)
            q.append(child)
    return root


def flatten_model_tree(tree: dict | None) -> list[str]:
    """Flatten a build_model_tree() into a list of dotted module names, so the agent can
    see valid targets for module-scoped actions (reset_module / freeze)."""
    if not tree:
        return []
    out: list[str] = []
    stack = [tree]
    while stack:
        node = stack.pop()
        name = node.get("name")
        if name and name != "Model":
            out.append(name)
        stack.extend(node.get("children", []))
    return out
