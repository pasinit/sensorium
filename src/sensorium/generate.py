"""Generate episodes from a config file, plus a manifest for reproducibility.

Usage:
    sensorium-generate configs/sample.toml

The config names an output directory, a master seed, and per body the
policies, episodes per policy, steps and simulator settings (see
``configs/sample.toml``). A body may instead hold several named *variants*
(``[humanoid.hand_seated]``, ``[humanoid.pincer_seated]``, ...), each with its
own policies, steps and settings, for bodies whose settings change what is
simulated (the humanoid's end effector and posture); every variant runs in its
own process. Every episode's seed is derived from
``(master_seed, body, variant, policy, index)`` (the variant is left out when
there is none), so any single episode can be regenerated on its own.

Each body runs in its own process (simulators don't share a process well:
e.g. MuJoCo's renderer fails to load after MetaDrive), so bodies also
generate in parallel.

``<out_dir>/manifest.json`` records the config, the code version (git commit,
dirty flag), package versions, platform, and per episode its seed, policy,
length and a content hash of its arrays. The content hash ignores zip
metadata, so two runs that produce identical data have identical hashes.
"""

import hashlib
import json
import logging
import multiprocessing
import platform
import subprocess
import sys
import time
import tomllib
from importlib import metadata
from pathlib import Path

import numpy as np

from sensorium.bodies import make_recorder
from sensorium.episode import Episode

log = logging.getLogger("sensorium.generate")

BODIES = ("car", "drone", "piano", "humanoid")
TRACKED_PACKAGES = (
    "numpy", "metadrive-simulator", "gym-pybullet-drones", "pybullet",
    "robopianist", "mujoco", "dm-control", "zarr",
)  # fmt: skip


def episode_seed(master_seed: int, body: str, policy: str, index: int, variant: str = "") -> int:
    """Deterministic 31-bit seed for one episode (stable across runs and machines)."""
    parts = [str(master_seed), body, *([variant] if variant else []), policy, str(index)]
    key = "/".join(parts).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big") & 0x7FFF_FFFF


def content_hash(episode: Episode) -> str:
    """SHA-256 over every array and the text, in a fixed order (first 16 hex chars)."""
    digest = hashlib.sha256()
    arrays = {
        "rgb": episode.rgb,
        "audio": episode.audio,
        "proprio": episode.proprio,
        "touch": episode.touch,
        "action": episode.action,
    }
    arrays |= {f"privileged/{k}": v for k, v in episode.privileged.items()}
    for name in sorted(arrays):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(arrays[name]).tobytes())
    digest.update("\n".join(episode.text).encode())
    return digest.hexdigest()[:16]


def provenance() -> dict[str, object]:
    """Code version, package versions and platform of this run."""

    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip()

    versions: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }


def record_body(
    body: str, spec: dict[str, object], master_seed: int, out_dir: Path, variant: str = ""
) -> list[dict[str, object]]:
    """Record and save every episode for one body (or one variant of it).

    Returns their manifest entries. Runs inside a worker process, one per body
    (or variant).
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")  # worker processes too
    settings = dict(spec.get("settings", {}))  # type: ignore[call-overload]
    entries: list[dict[str, object]] = []
    recorder = make_recorder(body, settings)
    try:
        for policy in spec["policies"]:  # type: ignore[attr-defined]
            for index in range(int(spec["episodes_per_policy"])):  # type: ignore[call-overload]
                seed = episode_seed(master_seed, body, policy, index, variant)
                started = time.time()
                episode = recorder.record(seed=seed, policy=policy, steps=int(spec["steps"]))  # type: ignore[call-overload]
                stem = "-".join([body, *([variant] if variant else []), policy])
                name = f"{body}/{stem}-{index:03d}.npz"
                episode.save(out_dir / name)
                entries.append(
                    {
                        "file": name,
                        "body": body,
                        "policy": policy,
                        **({"variant": variant} if variant else {}),
                        "index": index,
                        # Registry keys (docs/pools.md): what the episode is about and which
                        # specific content it has; the effector when the body has variants.
                        "task": episode.meta.get("task"),
                        "instance": episode.meta.get("instance", "default"),
                        **{k: episode.meta[k] for k in ("effector",) if k in episode.meta},
                        "seed": seed,
                        "steps": episode.length,
                        "end_reason": episode.meta.get("end_reason"),
                        "content_hash": content_hash(episode),
                    }
                )
                seconds = time.time() - started
                end = episode.meta.get("end_reason")
                log.info(
                    "[%s] %s: %d steps (%s) in %.1fs", body, name, episode.length, end, seconds
                )
    finally:
        recorder.close()
    return entries


def generate(config_path: Path) -> Path:
    """Generate every episode the config asks for; return the output directory.

    Raises:
        ValueError: if the config names an unknown body or policy.
    """
    raw = config_path.read_bytes()
    config = tomllib.loads(raw.decode())
    out_dir = Path(config["out_dir"])
    master_seed = int(config["master_seed"])
    bodies = [body for body in BODIES if body in config]
    jobs: list[tuple[str, dict[str, object], int, Path, str]] = []
    for body in bodies:
        if "policies" in config[body]:
            jobs.append((body, config[body], master_seed, out_dir, ""))
        else:  # a table of named variants
            jobs += [
                (body, spec, master_seed, out_dir, name) for name, spec in config[body].items()
            ]
    workers = min(len(jobs), int(config.get("workers", len(jobs))))  # top-level cap on processes
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        per_body = pool.starmap(record_body, jobs)
    manifest = {
        "config_file": str(config_path),
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "config": config,
        "provenance": provenance(),
        "episodes": [entry for entries in per_body for entry in entries],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log.info("wrote %d episodes + manifest.json to %s", len(manifest["episodes"]), out_dir)
    return out_dir


def main() -> None:
    """CLI entry point: ``sensorium-generate <config.toml>``."""
    if len(sys.argv) != 2:
        sys.exit("usage: sensorium-generate <config.toml>")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    generate(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
