"""Convert HumanoidBench's released ``reach_two_hands`` checkpoint to a numpy ``.npz``.

Usage: python scripts/convert_reach_policy.py [humanoid-bench dir]   (default
``<external>/humanoid-bench``; run by ``make h1-assets``)

Reads ``data/reach_two_hands/{torch_model.pt, mean.npy, var.npy}`` of the sparse HumanoidBench
checkout and writes ``reach_two_hands.npz`` next to it (``w1..w3`` of shape ``[out, in]``,
``b1..b3``, and the observation ``mean`` and ``var``: the files hold the same row 32,768
times, row 0 is used). The ``.pt`` file is read without torch: it is a zip archive of a
pickled ``OrderedDict`` whose tensors are rebuilt from raw storages by a restricted unpickler.
``scripts/verify_reach_policy.py`` checks the result against torch.
"""

import pickle
import sys
import zipfile
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

import numpy as np

from sensorium.bodies.h1_scene import external_dir

DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16}
LAYERS = {"dense1": "1", "dense2": "2", "dense3": "3"}


class StorageType:
    """Stand-in for ``torch.FloatStorage`` and friends: only the dtype matters."""

    def __init__(self, dtype: type[np.generic]) -> None:
        self.dtype = dtype


def rebuild_tensor(
    storage: np.ndarray,
    offset: int,
    size: tuple[int, ...],
    stride: tuple[int, ...],
    *_: object,
) -> np.ndarray:
    """Stand-in for ``torch._utils._rebuild_tensor_v2``: a strided view of the storage."""
    item = storage.itemsize
    return np.lib.stride_tricks.as_strided(
        storage[offset:], shape=size, strides=tuple(s * item for s in stride)
    ).copy()


class CheckpointUnpickler(pickle.Unpickler):
    """Unpickles a torch checkpoint's ``data.pkl`` into numpy arrays, refusing anything else."""

    def __init__(self, handle: zipfile.ZipExtFile, archive: zipfile.ZipFile, prefix: str) -> None:
        super().__init__(handle)
        self.archive, self.prefix = archive, prefix

    def find_class(self, module: str, name: str) -> Callable[..., object]:
        """Only the pieces of a plain state dict are allowed."""
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return rebuild_tensor
        if module == "torch" and name in DTYPES:
            return lambda: StorageType(DTYPES[name])
        raise pickle.UnpicklingError(f"not a plain state dict: {module}.{name}")

    def persistent_load(self, pid: tuple[object, ...]) -> np.ndarray:
        """A storage: ``("storage", type, key, location, size)`` -> its raw array."""
        kind, storage_type, key = pid[0], pid[1], str(pid[2])
        if kind != "storage":
            raise pickle.UnpicklingError(f"unknown persistent id {kind!r}")
        dtype = storage_type().dtype if callable(storage_type) else storage_type.dtype
        raw = self.archive.read(f"{self.prefix}/data/{key}")
        return np.frombuffer(raw, dtype=dtype)


def load_state_dict(path: Path) -> dict[str, np.ndarray]:
    """The tensors of a torch checkpoint (a plain state dict) as numpy arrays."""
    with zipfile.ZipFile(path) as archive:
        pkl = next(n for n in archive.namelist() if n.endswith("data.pkl"))
        with archive.open(pkl) as handle:
            state = CheckpointUnpickler(handle, archive, pkl.rsplit("/", 1)[0]).load()
    return {str(k): np.asarray(v) for k, v in state.items()}


def convert(humanoid_bench: Path) -> Path:
    """Write ``reach_two_hands.npz`` and return its path."""
    folder = humanoid_bench / "data" / "reach_two_hands"
    state = load_state_dict(folder / "torch_model.pt")
    arrays: dict[str, np.ndarray] = {}
    for layer, index in LAYERS.items():
        arrays[f"w{index}"] = state[f"{layer}.weight"].astype(np.float64)
        arrays[f"b{index}"] = state[f"{layer}.bias"].astype(np.float64)
    arrays["mean"] = np.load(folder / "mean.npy")[0].astype(np.float64)
    arrays["var"] = np.load(folder / "var.npy")[0].astype(np.float64)
    target = humanoid_bench / "reach_two_hands.npz"
    np.savez(target, **arrays)
    return target


def main() -> None:
    """Convert the checkpoint of the HumanoidBench checkout named on the command line."""
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else external_dir() / "humanoid-bench"
    sys.stdout.write(f"{convert(root)}\n")


if __name__ == "__main__":
    main()
