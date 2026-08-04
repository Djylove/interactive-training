"""Bounded loading for versioned embodied dataset manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from interactive_training.integrations.xpolicylab.contracts import DatasetManifest


def _stable_digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _verify_manifest_digests(payload: object) -> None:
    if not isinstance(payload, dict):
        raise TypeError("dataset manifest must be a JSON object")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("dataset manifest requires episodes")
    identities = []
    for index, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise TypeError(f"dataset episode {index} must be an object")
        files = episode.get("files")
        if episode.get("file_manifest_sha256") != _stable_digest(files):
            raise ValueError(f"dataset episode {index} file manifest digest mismatch")
        claimed_episode_digest = episode.get("episode_manifest_sha256")
        episode_payload = dict(episode)
        episode_payload.pop("episode_manifest_sha256", None)
        if claimed_episode_digest != _stable_digest(episode_payload):
            raise ValueError(f"dataset episode {index} manifest digest mismatch")
        identities.append(
            {
                "episode_id": episode.get("episode_id"),
                "episode_manifest_sha256": claimed_episode_digest,
            }
        )
    if payload.get("dataset_sha256") != _stable_digest(identities):
        raise ValueError("dataset identity digest mismatch")


def load_dataset_manifest(
    path: str | Path, *, max_bytes: int = 64 * 1024 * 1024
) -> DatasetManifest:
    source = Path(path).expanduser()
    if source.is_symlink():
        raise ValueError("dataset manifest cannot be a symlink")
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"dataset manifest not found: {source}")
    if source.stat().st_size > max_bytes:
        raise ValueError(f"dataset manifest exceeds {max_bytes} bytes")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid dataset manifest JSON: {source}: {exc}") from exc
    _verify_manifest_digests(payload)
    return DatasetManifest.model_validate(payload)


def verify_dataset_files(manifest: DatasetManifest) -> None:
    """Verify that every manifested episode file is unchanged and regular."""

    for episode in manifest.episodes:
        root = Path(episode.path).expanduser()
        if root.is_symlink():
            raise ValueError(f"dataset episode cannot be a symlink: {root}")
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"dataset episode not found: {root}")
        for artifact in episode.files:
            relative = Path(artifact.path)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\x00" in artifact.path
            ):
                raise ValueError(f"unsafe dataset artifact path: {artifact.path}")
            path = root / relative
            if path.is_symlink() or not path.is_file():
                raise FileNotFoundError(f"dataset artifact is not regular: {path}")
            if path.stat().st_size != artifact.size:
                raise ValueError(f"dataset artifact size mismatch: {path}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != artifact.sha256:
                raise ValueError(f"dataset artifact digest mismatch: {path}")


def validate_dataset_manifest_profile(
    manifest: DatasetManifest,
) -> DatasetManifest:
    """Dispatch to an explicit recorder/robot profile validator."""

    from interactive_training.integrations.xpolicylab.profiles.gr3 import (
        GR3_DAGGER_PROFILE_ID,
        validate_gr3_dagger_manifest,
    )

    if manifest.profile_id == GR3_DAGGER_PROFILE_ID:
        return validate_gr3_dagger_manifest(manifest)
    from interactive_training.integrations.xpolicylab.profiles.gr3_anygrasp import (
        GR3_ANYGRASP_PROFILE_ID,
        validate_gr3_anygrasp_manifest,
    )

    if manifest.profile_id == GR3_ANYGRASP_PROFILE_ID:
        return validate_gr3_anygrasp_manifest(manifest)
    from interactive_training.integrations.xpolicylab.profiles.robotwin import (
        ROBOTWIN_CLEAN_PROFILE_ID,
        validate_robotwin_clean_manifest,
    )

    if manifest.profile_id == ROBOTWIN_CLEAN_PROFILE_ID:
        return validate_robotwin_clean_manifest(manifest)
    raise ValueError(f"unsupported dataset profile: {manifest.profile_id}")
