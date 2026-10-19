# Pools (D0.1)

A **pool** is a frozen, ordered list of whole episodes for pretraining, built
from the episodes we have generated. It fixes the four things E1 varies and
holds everything else equal: the **senses** the model sees, the **kind of
experience** (expert, noisy expert, babbling), the **number of sources**, and
the **end-effector ratio**, all at the same number of steps. Pure Python over
the existing episode format: no simulator, no GPU.

```
manifest.json (sensorium-generate)  ->  index.parquet  ->  pool.json
        sensorium-index                  sensorium-pool
configs/sources.toml  (registry: body, task -> source, effector, kind)
configs/splits/v3.toml (frozen split: pretrain vs held-out sources)
```

```sh
make index        # data/sample/manifest.json -> data/sample/index.parquet
make pool         # configs/pools/example.toml -> data/sample/pool.json
```

Override with `make index MANIFESTS="a/manifest.json b/manifest.json" INDEX=...`
and `make pool POOL_SPEC=... POOL=...`.

## Sources

An **experience source** is one `(body, effector, task)` triple, defined in
`configs/sources.toml` (`sensorium.registry`):

| Field | Meaning |
|---|---|
| `name` | unique id, e.g. `piano/rp1m`; what splits and pools refer to |
| `body`, `task` | how a manifest entry is matched (`task` defaults per body, see below) |
| `effector` | `hand`, `pincer`, `none` or `alien` (car, drone) |
| `family` | group of related sources, e.g. `piano` |
| `tags` | exactly one of `contact_rich` / `vision_dominated`, plus anything else |

The same file maps the recorded `policy` name to the **kind** of experience in
a `[policy_kinds]` table: `expert` to expert, `noisy` to noisy, `explore` to
babbling. An unknown `(body, task)` or policy in a manifest is an error that
lists what is registered.

### Tasks and instances

Every episode has `body`, `effector`, `task`, `instance` and `policy` (which
gives `kind`). A source stays `(body, effector, task)`. The **instance** is a
free string naming the specific content of the task: a song id for `piano`, a
mocap clip id for a walk, a demo id for a BiGym task, an object-layout id for
a play task. Splits can hold out whole tasks or instances within a task.

**Manifest fields** (all written by `sensorium.generate`): `file` (relative
to the manifest), `body`, `policy`, `seed`, `steps`, `content_hash`, and the
registry keys `task`, `instance` and, for bodies with end-effector variants,
`effector`, copied from `episode.meta`, where each recorder sets them (car:
`drive` / `map-<seed>`; drone: `waypoints` / `course-<seed>`; piano: `rp1m` /
the RP1M song id; humanoid: `stand`, `table` or `stand_table` / `layout-<seed>`). Manifests
that predate these fields still index:

| Field | When absent |
|---|---|
| `task` | the body's `[default_task]` (`car` drive, `drone` waypoints, `piano` rp1m) |
| `instance` | the string `"default"` |

A body with several sources for one task (the humanoid's `table` or `stand_table` with hands or
pincers) is resolved by `effector`; without it the lookup is ambiguous and
`sensorium-index` fails with the registered triples listed.

### Adding a source

1. Add a `[[source]]` to `configs/sources.toml` (and, for a new body, its
   `[default_task]`; for a new policy name, `[policy_kinds]`).
2. Assign it to exactly one section of the split file (`configs/splits/v3.toml`).
   `load_splits` fails if any registry source is unassigned.
3. Regenerate the index (`make index`).

## Splits

`configs/splits/v3.toml` is frozen before training. Changelog: **v3** (2026-10-15) adds the experiment sources (`humanoid/{hand,pincer}/exp_*`) and holds out
the few-shot tasks T1-T4 and BiGym's StackBlocks and DrawerTopOpen; sources now list `programs` and `props`, and a
split fails to load if a pretraining source uses a held-out one ([experiments.md](experiments.md)); v2 no longer loads
against the registry. **v2** (2026-10-14) gives
`stand_supported` episodes their own sources, `humanoid/{hand,pincer}/stand_table`, instead of
sharing `table` with the seated ones (no data existed at scale; there was no v1; `v0.toml` is kept
for the record and no longer loads against the registry); **v0** was the first split. `[pretrain] sources`
lists the sources pools draw from, by name. Two kinds of held-out content:

```toml
[heldout.tasks]        # whole sources, by source name
"bigym/drawer" = { reason = "pincer side, k-shot target", tag = "contact_rich" }

[heldout.instances.rp1m]   # instances within a task, keyed by task name
ids = ["song_0123", "song_0456"]
reason = "RP1M out-of-distribution songs, hand side"
tag = "contact_rich"
```

Every held-out entry needs a `reason` and a `tag` (`contact_rich` or
`vision_dominated`, reported per side). `load_splits` checks that every registry
source is in exactly one section, that each instance list names a registered
task and has ids, and that no instance is listed under a task that is itself
held out. The rest of an instance-held-out task stays in the pretraining pool.

`build_pool` leaves held-out instances out, and `validate_pool` asserts that
neither a held-out task nor a held-out instance is in the pool. The pool JSON
records, per task, how many held-out instances were present and excluded.
Each pool also records the split file's hash.

v0 holds nothing out yet. **TODO (D0.5):** generate `[heldout.instances.rp1m]`
from RP1M's own in-distribution / out-of-distribution song lists with a small
script (not written), once the piano recorder writes the song as `instance`.

## The pool spec

```toml
name = "mix-8"
total_steps = 1_000_000
senses = ["rgb", "audio_logmel", "proprio", "touch", "action"]   # subset; action is always kept
seed = 0
n_sources = 8                # int, or "all"
split = "pretrain"           # section of the split file
tolerance = 0.02             # optional, default 2%

[kinds]                      # relative weights
expert = 1
noisy = 1
babbling = 1                 # "babbling only" = just this line

[effector_ratio]             # relative weights; absent or 0 = excluded
hand = 2
pincer = 2
none = 1                     # alien bodies (car, drone) only if "alien" > 0
```

### How it is built

1. **Allowed episodes:** episodes of the sources in the split's section whose
   effector has a positive weight, minus held-out instances.
2. **Nesting:** rank those sources by `sha256(f"{seed}/{source}")` and take the
   first `n_sources`. The ranking depends only on the seed, so n=8 is inside
   n=16 is inside all. Caveat: the ranking is global, so a small n can miss an
   effector entirely; the build then fails (next point) instead of quietly
   changing the ratio.
3. **Strata:** every `(effector, kind)` cell gets
   `total_steps x effector_weight x kind_weight` steps. Episodes of the chosen
   sources are ordered deterministically (round-robin over sources, hash-shuffled
   within) and taken whole, first-fit, until the cell is within `tolerance` of its
   target.
4. **Checks (fail loudly, `PoolError`):** every cell and the total must be within
   `tolerance`; the message names the cell and how many steps were available. No
   padding, no truncation, no silent re-weighting.

Everything is deterministic: the same index, spec and split file give the same
episode order and the same pool hash (`sha256` of the senses and the ordered
episode content hashes, so it does not depend on file locations).

### Output

`pool.json` records the spec, seed, split version and hash, the chosen
sources, per-stratum target and actual steps, the total, the pool content hash
the excluded held-out instance counts per task and the ordered episodes (path, source,
task, instance, kind, effector, steps, content hash).
Training code reads it with `sensorium.pool.read_pool` and loads episodes with

```python
inputs = model_inputs(Episode.load(path), senses=pool.senses)
```

which drops every sense not in the pool's subset; `action` is always kept, and
`privileged`/`text` can never appear.
