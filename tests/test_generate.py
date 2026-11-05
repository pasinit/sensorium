"""The generator's per-episode wall-clock guard and its partial manifests."""

import json
import time
from pathlib import Path

import numpy as np
import pytest

from sensorium import generate
from sensorium.episode import Episode


class FakeEpisode:
    """Just what ``record_body`` uses of an episode."""

    length = 3
    meta: dict[str, object] = {"end_reason": "steps"}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("episode")


class FakeRecorder:
    """Hangs on its first episode, records the others at once."""

    made = 0

    def __init__(self) -> None:
        FakeRecorder.made += 1
        self.calls = 0

    def record(self, seed: int, policy: str, steps: int) -> FakeEpisode:
        self.calls += 1
        if FakeRecorder.made == 1 and self.calls == 1:
            time.sleep(30)  # stuck
        return FakeEpisode()

    def close(self) -> None:
        pass


def test_a_stuck_episode_is_skipped_and_the_rest_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(generate, "make_recorder", lambda body, settings: FakeRecorder())
    monkeypatch.setattr(generate, "content_hash", lambda episode: "hash")
    spec = {"policies": ["hold"], "episodes_per_policy": 3, "steps": 3, "episode_timeout_s": 1}
    started = time.time()
    entries = generate.record_body("humanoid", spec, 0, tmp_path, "hand_x")
    assert time.time() - started < 10
    assert [e["index"] for e in entries] == [1, 2]  # episode 0 hung and was skipped
    assert FakeRecorder.made == 2  # the recorder was rebuilt after the stuck episode
    parts = tmp_path / "manifest.parts" / "humanoid-hand_x.jsonl"
    lines = [json.loads(line) for line in parts.read_text().splitlines()]
    assert [line["file"] for line in lines] == [
        "humanoid/humanoid-hand_x-hold-001.npz",
        "humanoid/humanoid-hand_x-hold-002.npz",
    ]


def tiny_episode(seed: int) -> Episode:
    """A valid two-step episode whose content depends on the seed."""
    rng = np.random.default_rng(seed)
    return Episode(
        body="humanoid",
        dt=0.05,
        rgb=np.zeros((2, 256, 256, 3), dtype=np.uint8),
        audio=np.zeros((2, 800), dtype=np.int16),
        audio_logmel=np.zeros((2, 0), dtype=np.float16),
        proprio=rng.normal(size=(2, 2)).astype(np.float32),
        touch=np.zeros((2, 0), dtype=np.float32),
        action=np.zeros((2, 1), dtype=np.float32),
        proprio_names=["q_a", "q_b"],
        touch_names=[],
        action_names=["a"],
        text=["", ""],
        meta={"end_reason": "steps", "seed": seed},
    )


class CountingRecorder:
    """Records tiny real episodes and counts the calls."""

    calls: list[int] = []

    def record(self, seed: int, policy: str, steps: int) -> Episode:
        CountingRecorder.calls.append(seed)
        return tiny_episode(seed)

    def close(self) -> None:
        pass


def test_skip_existing_resumes_a_stopped_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(generate, "make_recorder", lambda body, settings: CountingRecorder())
    spec = {"policies": ["hold"], "episodes_per_policy": 4, "steps": 2, "skip_existing": True}
    first = generate.record_body("humanoid", spec, 7, tmp_path, "hand_x")
    assert len(CountingRecorder.calls) == 4
    names = [str(e["file"]) for e in first]
    parts = tmp_path / "manifest.parts" / "humanoid-hand_x.jsonl"
    # a stop: episode 1 never written, episode 2 saved but not in the manifest, episode 3 damaged
    (tmp_path / names[1]).unlink()
    lines = parts.read_text().splitlines()
    parts.write_text("\n".join([lines[0], lines[3][:20]]) + "\n")  # the last line cut short
    (tmp_path / names[3]).write_bytes(b"not an npz")
    CountingRecorder.calls = []
    second = generate.record_body("humanoid", spec, 7, tmp_path, "hand_x")
    seeds = [int(e["seed"]) for e in first]  # type: ignore[call-overload]
    assert CountingRecorder.calls == [seeds[1], seeds[3]]  # only the missing and the damaged ones
    assert [e["file"] for e in second] == names
    assert [e["seed"] for e in second] == seeds and second[2]["content_hash"] == first[2][
        "content_hash"
    ]
    assert not list(tmp_path.rglob("*.partial.npz"))  # every save was renamed into place
    CountingRecorder.calls = []
    generate.record_body("humanoid", {**spec, "skip_existing": False}, 7, tmp_path, "hand_x")
    assert len(CountingRecorder.calls) == 4  # off: everything recorded again
