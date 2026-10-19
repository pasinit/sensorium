"""Run the GPU smoke test of the sensory layer on Modal. PREPARED, NOT RUN.

Needs a Modal account with a payment method on file for GPUs (Modal docs, "GPU acceleration")
and the Modal CLI; nothing else of this project. From the repo root, after ``make sample-h1``:

    uvx modal run scripts/modal_gpu_smoke.py                    # H100, 15 minutes, batch 128
    SMOKE_GPU=A100-80GB uvx modal run scripts/modal_gpu_smoke.py --minutes 10

``H100!`` (the default) opts out of Modal's silent upgrade to an H200 so the numbers are an
H100's. The report is written to ``runs/gpu-smoke/report.json`` and printed. The image holds
torch from PyPI (bundles its CUDA libraries; Modal hosts offer driver CUDA 13.0, so check
``hardware.cuda`` in the report and, if CUDA is unavailable, pin a cu12x wheel); the package,
``configs/`` and the 8-episode H1 sample are attached at start-up, not baked in.

Cost, Modal list prices (modal.com/pricing, 2026-10-14): H100 $0.001097/s, A100 80GB
$0.000694/s, CPU $0.0000131 per core-s, memory $0.00000222 per GiB-s. A 25-minute run with 8
cores and 32 GiB costs about $1.9 on an H100 and $1.3 on an A100 80GB; the Starter plan
includes $30 of credit a month.
"""

import json
import os
from pathlib import Path

import modal

GPU = os.environ.get("SMOKE_GPU", "H100!")
TIMEOUT_MIN = 45

image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install(
        "torch==2.14.1", "numpy>=2.0", "pillow>=10", "pyarrow>=17", "imageio-ffmpeg>=0.5"
    )
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir("src/sensorium", "/root/src/sensorium")
    .add_local_dir("configs", "/root/configs")
    .add_local_dir("data/sample_h1", "/root/data/sample_h1")
)

app = modal.App("sensorium-gpu-smoke")


@app.function(image=image, gpu=GPU, cpu=8.0, memory=32768, timeout=TIMEOUT_MIN * 60)
def smoke(minutes: float, batch_size: int, workers: list[int]) -> dict[str, object]:
    """Run ``sensorium.model.gpu_smoke`` in the container and return its report."""
    os.chdir("/root")
    from sensorium.model import gpu_smoke  # noqa: PLC0415  (only exists in the image)

    return gpu_smoke.run(
        [
            "configs/model/v0.toml",
            "--paths",
            "data/sample_h1/humanoid/*.npz",
            "--minutes",
            str(minutes),
            "--batch-size",
            str(batch_size),
            "--workers",
            *map(str, workers),
            "--out",
            "runs/gpu-smoke",
        ]
    )


@app.local_entrypoint()
def main(minutes: float = 15.0, batch_size: int = 128, workers: str = "4,8") -> None:
    """Launch the smoke test and save its report locally."""
    report = smoke.remote(minutes, batch_size, [int(w) for w in workers.split(",")])
    out = Path("runs/gpu-smoke")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))  # noqa: T201
