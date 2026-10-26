"""The Push-T converter (``sensorium.pusht``, ``scripts/pusht_convert.py``) on a tiny h5."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py", reason="needs the pusht extra: make setup")
torch = pytest.importorskip("torch", reason="needs the model extra: make setup")

from sensorium import pusht  # noqa: E402
from sensorium.episode import Episode  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LENGTHS = (25, 31, 22)


def make_source(path: Path, lengths: tuple[int, ...] = LENGTHS, action_max: float = 1.0) -> None:
    """A stable-worldmodel HDF5 file with random columns, as the LeWM data is laid out."""
    rng = np.random.default_rng(0)
    n = sum(lengths)
    with h5py.File(path, "w") as f:
        f["pixels"] = rng.integers(0, 256, (n, 224, 224, 3), dtype=np.uint8)
        f["action"] = rng.uniform(-action_max, action_max, (n, 2)).astype(np.float32)
        f["proprio"] = rng.uniform(0, 512, (n, 4)).astype(np.float32)
        f["state"] = rng.uniform(0, 512, (n, 7)).astype(np.float32)
        f["ep_len"] = np.array(lengths, dtype=np.int32)
        f["ep_offset"] = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)


def test_an_episode_has_vision_proprio_and_actions_and_no_touch_or_audio(tmp_path: Path) -> None:
    src = tmp_path / "src.h5"
    make_source(src, action_max=1.2)
    with h5py.File(src, "r") as f:
        arrays = pusht.read_episode(f, 1)
    ep = pusht.to_episode(1, arrays, commit="abc1234")
    t = LENGTHS[1]
    assert ep.body == "pusht" and ep.dt == pytest.approx(0.1) and ep.length == t
    assert ep.rgb.shape == (t, 256, 256, 3)
    assert ep.proprio_names == pusht.PROPRIO_NAMES and ep.action_names == pusht.ACTION_NAMES
    assert ep.touch.shape == (t, 0) and ep.audio_logmel.shape == (t, 0) and not ep.audio.any()
    np.testing.assert_allclose(ep.action, np.clip(arrays["action"], -1, 1))  # stored as they are
    assert ep.meta["actions_clipped"] == int((np.abs(arrays["action"]) > 1).sum()) > 0
    np.testing.assert_array_equal(ep.privileged["state"], arrays["state"])
    np.testing.assert_array_equal(ep.privileged["block_pose"], arrays["state"][:, 2:5])
    assert ep.meta["source_episode"] == 1 and ep.meta["licence"] == "MIT"
    assert ep.meta["code_commit"] == "abc1234" and "resize" in str(ep.meta["lewm_frames"])
    ep.save(tmp_path / "ep.npz")
    back = Episode.load(tmp_path / "ep.npz")
    assert back.rgb.shape == ep.rgb.shape and np.abs(back.rgb.astype(int) - ep.rgb).mean() < 8
    with pytest.raises(ValueError, match="action"):
        pusht.to_episode(1, {**arrays, "action": arrays["action"][:, :1]})


def test_the_clip_split_is_torchs_random_split_of_stable_worldmodels_clips() -> None:
    lengths = [25, 31, 19, 40]  # 19 < span 20: no clip
    split = pusht.clip_split(lengths)
    every = pusht.clips(lengths, pusht.LEWM_NUM_STEPS, pusht.LEWM_FRAMESKIP)
    assert len(every) == (25 - 19) + (31 - 19) + (40 - 19)
    assert all(ep != 2 for ep, _ in every)
    expected = torch.utils.data.random_split(
        range(len(every)), [0.9, 0.1], generator=torch.Generator().manual_seed(3072)
    )
    assert split["train"] == [every[i] for i in expected[0].indices]
    assert split["val"] == [every[i] for i in expected[1].indices]
    assert pusht.split_lengths(10, [0.9, 0.1]) == [9, 1]
    assert pusht.split_lengths(7, [0.9, 0.1]) == [7, 0]


def test_the_converter_writes_episodes_and_lewms_h5_with_the_same_pixels(tmp_path: Path) -> None:
    src, out = tmp_path / "src.h5", tmp_path / "out"
    make_source(src)

    def run(*args: str) -> None:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/pusht_convert.py"), *args],
            check=True,
            capture_output=True,
        )

    run("stats", str(src), str(out / "stats.json"))
    info = json.loads((out / "stats.json").read_text())
    assert info["episodes"] == 3 and info["lengths"] == list(LENGTHS)
    for shard in ("0/2", "1/2"):
        run("episodes", str(src), str(out), "--shard", shard, "--commit", "abc1234")
    files = sorted((out / "pusht").glob("*.npz"))
    assert [f.name for f in files] == ["pusht-00000.npz", "pusht-00001.npz", "pusht-00002.npz"]
    for shard in ("0/2", "1/2"):
        run("lewm", str(src), str(out), str(out / f"part{shard[0]}.h5"), "--shard", shard)
    run(
        "merge",
        str(out / "lewm.h5"),
        str(out / "part0.h5"),
        str(out / "part1.h5"),
        "--commit",
        "abc1234",
    )
    with h5py.File(out / "lewm.h5", "r") as f, h5py.File(src, "r") as s:
        assert f["ep_len"][:].tolist() == list(LENGTHS) and f.attrs["code_commit"] == "abc1234"
        assert f["pixels"].shape == (sum(LENGTHS), 64, 64, 3)
        np.testing.assert_array_equal(f["action"][:], s["action"][:])  # LeWM sees its own units
        for index, path in enumerate(files):
            ours = pusht.lewm_frames(Episode.load(path, privileged=False))
            np.testing.assert_array_equal(pusht.read_episode(f, index)["pixels"], ours)
    run("split", str(src), str(out / "split.json"))
    split = json.loads((out / "split.json").read_text())
    n_clips = sum(length - 19 for length in LENGTHS if length >= 20)
    assert len(split["lewm_clips"]["train"]) + len(split["lewm_clips"]["val"]) == n_clips
    run("select", str(out / "lewm.h5"), str(out / "sample.h5"), "--first", "2")
    with h5py.File(out / "sample.h5", "r") as f:
        assert f["ep_len"][:].tolist() == list(LENGTHS[:2]) and f.attrs["code_commit"] == "abc1234"
    run("subset", str(src), str(out / "subset.json"), "--count", "2", "--seed", "0")
    chosen = json.loads((out / "subset.json").read_text())
    assert chosen == np.sort(np.random.default_rng(0).choice(3, 2, replace=False)).tolist()
    run("select", str(out / "lewm.h5"), str(out / "sub.h5"), "--episodes", str(out / "subset.json"))
    run("split", str(src), str(out / "sub_split.json"), "--episodes", str(out / "subset.json"))
    sub_split = json.loads((out / "sub_split.json").read_text())
    with h5py.File(out / "sub.h5", "r") as f:
        sub_lengths = f["ep_len"][:].tolist()
    assert sub_lengths == [LENGTHS[i] for i in chosen]
    assert sub_split["lewm_clips"] == {
        k: [list(c) for c in v] for k, v in pusht.clip_split(sub_lengths).items()
    }
    assert all(
        chosen[e] == src_e
        for (e, _), (src_e, _) in zip(
            sub_split["lewm_clips"]["train"], sub_split["source_clips"]["train"], strict=True
        )
    )
    run("report", str(src), str(out), str(out / "stats.json"), str(out / "fig"))
    assert (out / "fig" / "pusht_strip_224_vs_64.png").exists()
