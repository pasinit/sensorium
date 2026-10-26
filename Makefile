# Kit-managed dev targets (ruff config in pyproject.toml, mypy in .claude/mypy.ini),
# plus project targets for the E1 data sample.
.PHONY: setup rp1m-toy h1-assets a1-gate lint format typecheck test check smoke sample browse sample-h1 index pool pretrain-smoke measure

RP1M_TOY_URL    := https://edmond.mpg.de/api/access/datafile/262025
RP1M_TOY_SHA256 := 20048c4abf13bc7d6f9eac3674e02a95b946d825dd767cc2b50140c3eae19dfc
RP1M_DIR        := data/external/rp1m_toy

# H1 humanoid assets (MJCF + meshes only, no humanoid-bench Python dependency), pinned to commits.
# Fetched sparsely (only the directories we load) into the ignored data/external/.
HUMANOID_BENCH_URL := https://github.com/carlosferrazza/humanoid-bench
HUMANOID_BENCH_SHA := cb1189039151c8aadaaa987b442da54383c87fab
MENAGERIE_URL      := https://github.com/google-deepmind/mujoco_menagerie
MENAGERIE_SHA      := 0059d4335f8156206f63a35662313385f7ad6d74

setup:
	# pybullet builds from source on macOS; its bundled zlib clashes with recent SDKs.
	CFLAGS="-Dfdopen=fdopen" uv sync --extra sims --extra model --extra pusht

# RP1M expert piano data, toy subset (351 MB, CC BY 4.0, doi:10.17617/3.XCE8NX), checksum-verified.
rp1m-toy:
	mkdir -p $(RP1M_DIR)
	test -f $(RP1M_DIR)/rp1m_toy.zip || curl -L --fail -o $(RP1M_DIR)/rp1m_toy.zip $(RP1M_TOY_URL)
	echo "$(RP1M_TOY_SHA256)  $(RP1M_DIR)/rp1m_toy.zip" | shasum -a 256 -c
	unzip -q -o $(RP1M_DIR)/rp1m_toy.zip -d $(RP1M_DIR)

# H1 humanoid: HumanoidBench assets (MIT, ~110 MB) and MuJoCo Menagerie's Robotiq 2F-85 (BSD-2, ~7 MB).
h1-assets:
	scripts/fetch_pinned.sh $(HUMANOID_BENCH_URL) $(HUMANOID_BENCH_SHA) data/external/humanoid-bench \
		humanoid_bench/assets/common humanoid_bench/assets/h1 humanoid_bench/assets/robots \
		humanoid_bench/assets/shadow_hand_submodel data/reach_two_hands
	uv run python scripts/convert_reach_policy.py
	scripts/fetch_pinned.sh $(MENAGERIE_URL) $(MENAGERIE_SHA) data/external/mujoco_menagerie robotiq_2f85

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy --config-file .claude/mypy.ini src tests

test:
	uv run pytest

check: lint typecheck test

# One tiny episode per body, end to end (~1 min).
smoke:
	uv run sensorium-generate configs/smoke.toml
	uv run sensorium-browse data/smoke

# Generate the review sample (CPU only) and build the browsable HTML index.
sample:
	uv run sensorium-generate configs/sample.toml

browse:
	uv run sensorium-browse data/sample

# The first humanoid review sample (H1 with Shadow hands and Robotiq pincers), then its pages.
sample-h1:
	uv run sensorium-generate configs/sample_h1.toml
	uv run python scripts/export_skin_layout.py data/sample_h1
	uv run sensorium-browse data/sample_h1

# Pass/fail lines for forces, contact, lifts and visibility of a babbling run; run it on every
# regeneration. Generate first: uv run sensorium-generate configs/a1.toml
A1_DIR ?= data/a1

a1-gate:
	uv run python -m sensorium.a1_gate $(A1_DIR)

# Episode index (parquet) over a generated sample, then an example pool from it.
# Override: make index MANIFESTS="a/manifest.json b/manifest.json" INDEX=data/all/index.parquet
MANIFESTS ?= data/sample/manifest.json
INDEX     ?= data/sample/index.parquet
POOL_SPEC ?= configs/pools/example.toml
POOL      ?= data/sample/pool.json

index:
	uv run sensorium-index $(INDEX) $(MANIFESTS)

pool:
	uv run sensorium-pool $(INDEX) $(POOL_SPEC) $(POOL)

# The sensory layer (Track B, docs/model.md): a 300-step CPU run of the tiny model on the
# H1 sample (make sample-h1 first), and the parameter / FLOP / speed table of the v0 config.
pretrain-smoke:
	uv run sensorium-pretrain configs/model/cpu_tiny.toml

measure:
	uv run python -m sensorium.model.measure configs/model/v0.toml --paths 'data/sample_h1/humanoid/*.npz'
