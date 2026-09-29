"""Content-addressed results.

A result is stored under a key that hashes everything that determines it (for a study: the study
key, the model key and the output name). Identical results made on different machines get the same
key, so result folders can be merged by copying files.
"""

import json
from pathlib import Path

import numpy as np
import xarray as xr


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _paths(self, key: str) -> tuple[Path, Path]:
        d = self.root / key[:2]
        return d / f"{key}.npy", d / f"{key}.json"

    def get(self, key: str) -> xr.DataArray | None:
        values, meta = self._paths(key)
        if not (values.exists() and meta.exists()):
            return None
        m = json.loads(meta.read_text(encoding="utf-8"))
        return xr.DataArray(np.load(values), dims=m["dims"], coords=m["coords"], name=m["name"])

    def put(self, key: str, da: xr.DataArray, provenance: dict) -> None:
        values, meta = self._paths(key)
        values.parent.mkdir(parents=True, exist_ok=True)
        tmp = values.with_suffix(".tmp.npy")
        np.save(tmp, np.asarray(da.values))
        tmp.replace(values)
        coords = {c: np.asarray(da.coords[c].values).tolist() for c in da.coords if c in da.dims}
        meta.write_text(json.dumps({"name": da.name, "dims": list(da.dims), "coords": coords,
                                    "provenance": provenance}), encoding="utf-8")
