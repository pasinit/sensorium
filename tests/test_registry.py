"""Source registry: the shipped TOML is valid, bad registries and unknown sources fail loudly."""

from pathlib import Path

import pytest

from sensorium.pool import PoolError, load_splits
from sensorium.registry import KINDS, load_registry

ROOT = Path(__file__).parent.parent
EXPERIMENT_SOURCES = {
    f"humanoid/{e}/{t}"
    for e in ("hand", "pincer")
    for t in (
        "exp_push exp_topple exp_slide exp_press exp_knock exp_door exp_turn "
        "push_to_target turn_reach"
    ).split()
} | {
    "humanoid/hand/exp_lift_drop",
    "humanoid/pincer/exp_lift_drop",
    "humanoid/pincer/stack",
    "humanoid/pincer/drawer_open",
    "bigym/pincer/stack_blocks",
    "bigym/pincer/drawer_top_open",
}
HELD_OUT = {
    "humanoid/pincer/stack",
    "humanoid/pincer/drawer_open",
    "humanoid/hand/push_to_target",
    "humanoid/pincer/push_to_target",
    "humanoid/hand/turn_reach",
    "humanoid/pincer/turn_reach",
    "bigym/pincer/stack_blocks",
    "bigym/pincer/drawer_top_open",
}
GOOD = """
[policy_kinds]
expert = "expert"

[[source]]
name = "x/y"
body = "x"
effector = "hand"
task = "y"
family = "f"
tags = ["contact_rich"]
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "sources.toml"
    path.write_text(text)
    return path


def test_shipped_registry_and_split_are_valid() -> None:
    registry = load_registry(ROOT / "configs/sources.toml")
    assert {s.name for s in registry.sources} == {
        "car/drive",
        "drone/waypoints",
        "piano/rp1m",
        "humanoid/hand/stand",
        "humanoid/hand/table",
        "humanoid/pincer/stand",
        "humanoid/pincer/table",
        "humanoid/hand/stand_table",
        "humanoid/pincer/stand_table",
        *EXPERIMENT_SOURCES,
    }
    assert registry.by_name("piano/rp1m").effector == "hand"
    assert registry.by_name("car/drive").effector == "alien"
    assert "contact_rich" in registry.by_name("piano/rp1m").tags
    assert {registry.kind_of(p) for p in ("expert", "noisy", "explore", "experiment")} == set(KINDS)
    assert registry.kind_of("experiment") == "experiment" != registry.kind_of("seated_babble")
    splits = load_splits(ROOT / "configs/splits/v3.toml", registry)
    assert set(splits.sections["pretrain"]) | set(splits.heldout) == {
        s.name for s in registry.sources
    }
    assert splits.version == "v3"
    assert registry.resolve("humanoid", "stand_table", "hand").name == "humanoid/hand/stand_table"


def test_resolve_uses_default_task_and_rejects_unknown() -> None:
    registry = load_registry(ROOT / "configs/sources.toml")
    assert registry.resolve("drone", None).name == "drone/waypoints"
    assert registry.resolve("drone", "waypoints").name == "drone/waypoints"
    with pytest.raises(ValueError, match="no registered source"):
        registry.resolve("humanoid", None)
    with pytest.raises(ValueError, match="no registered source"):
        registry.resolve("car", "park")


def test_resolve_needs_the_effector_when_a_task_has_variants() -> None:
    registry = load_registry(ROOT / "configs/sources.toml")
    assert registry.resolve("humanoid", "table", "pincer").name == "humanoid/pincer/table"
    with pytest.raises(ValueError, match="ambiguous"):
        registry.resolve("humanoid", "table")


def test_unknown_policy_is_rejected() -> None:
    registry = load_registry(ROOT / "configs/sources.toml")
    with pytest.raises(ValueError, match="unknown policy 'dance'"):
        registry.kind_of("dance")


def test_good_registry_loads(tmp_path: Path) -> None:
    assert load_registry(write(tmp_path, GOOD)).by_name("x/y").body == "x"


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('effector = "hand"', 'effector = "claw"', "effector"),
        ('tags = ["contact_rich"]', 'tags = ["shiny"]', "exactly one"),
        ('tags = ["contact_rich"]', 'tags = ["contact_rich", "vision_dominated"]', "exactly one"),
        ('family = "f"\n', "", "missing fields"),
        ('expert = "expert"', 'expert = "genius"', "kind"),
    ],
)
def test_invalid_registry_rejected(tmp_path: Path, old: str, new: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_registry(write(tmp_path, GOOD.replace(old, new)))


def test_duplicate_source_rejected(tmp_path: Path) -> None:
    second = GOOD.split("[[source]]")[1]
    with pytest.raises(ValueError, match="duplicate"):
        load_registry(write(tmp_path, GOOD + "\n[[source]]" + second))


def test_default_task_must_be_a_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="default task"):
        load_registry(write(tmp_path, GOOD + '\n[default_task]\nx = "nope"\n'))


def test_held_out_tasks_are_the_decided_ones_and_do_not_leak() -> None:
    """T1-T4 and BiGym's stack and drawer are held out; no pretraining source runs their
    program or uses their props (the leak check runs on load)."""
    registry = load_registry(ROOT / "configs/sources.toml")
    splits = load_splits(ROOT / "configs/splits/v3.toml", registry)
    assert set(splits.heldout) == HELD_OUT
    held = [registry.by_name(n) for n in splits.heldout]
    pretrain = [registry.by_name(n) for n in splits.sections["pretrain"]]
    held_programs = {p for s in held for p in s.programs}
    held_props = {p for s in held for p in s.props}
    assert held_programs == {"stack", "drawer_open", "push_to_target", "turn_reach"}
    assert not held_programs & {p for s in pretrain for p in s.programs}
    assert not held_props & {p for s in pretrain for p in s.props}
    # the neighbouring skills stay in pretraining
    programs = {p for s in pretrain for p in s.programs}
    assert {"push", "lift_drop", "door", "turn", "topple", "slide", "press", "knock"} <= programs


def test_leak_of_a_held_out_program_or_prop_is_rejected(tmp_path: Path) -> None:
    registry_text = """
[policy_kinds]
expert = "expert"

[[source]]
name = "a/seen"
body = "a"
effector = "hand"
task = "seen"
family = "f"
tags = ["contact_rich"]
programs = ["push"]
props = ["PROPS"]

[[source]]
name = "a/unseen"
body = "a"
effector = "hand"
task = "unseen"
family = "f"
tags = ["contact_rich"]
programs = ["PROGRAM"]
props = ["drawer"]
"""
    split_text = """
version = "t"
[pretrain]
sources = ["a/seen"]
[heldout.tasks]
"a/unseen" = { reason = "held out", tag = "contact_rich" }
"""
    path = tmp_path / "splits.toml"
    path.write_text(split_text)
    clean = write(tmp_path, registry_text.replace("PROPS", "objects").replace("PROGRAM", "open"))
    assert load_splits(path, load_registry(clean)).heldout == ("a/unseen",)
    by_prop = write(tmp_path, registry_text.replace("PROPS", "drawer").replace("PROGRAM", "open"))
    with pytest.raises(PoolError, match="prop 'drawer'"):
        load_splits(path, load_registry(by_prop))
    by_program = write(
        tmp_path, registry_text.replace("PROPS", "objects").replace("PROGRAM", "push")
    )
    with pytest.raises(PoolError, match="program 'push'"):
        load_splits(path, load_registry(by_program))


def test_experiment_is_its_own_stratum_in_a_pool_spec() -> None:
    """Pools can ask for experiment episodes alone, or in a ratio with babbling and expert."""
    from sensorium.pool import normalise  # noqa: PLC0415

    assert normalise({"experiment": 1.0}, KINDS) == {"experiment": 1.0}
    mixed = normalise({"expert": 1.0, "babbling": 1.0, "experiment": 2.0}, KINDS)
    assert mixed == {"expert": 0.25, "babbling": 0.25, "experiment": 0.5}
