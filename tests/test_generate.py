"""The generator's per-episode wall-clock guard and its partial manifests."""

import json
import time
from pathlib import Path

import pytest

from sensorium import generate


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
