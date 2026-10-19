"""MineLib instances as CSV files: downloaded once from https://minelib.org (CC BY-SA 3.0) into ``data/``.

``load(name)`` returns the paths of ``<name>.csv`` (MineLib's ``.blocks`` file with the column names of its
documentation, ``ID, IX, IY, IZ, ...``, plus ``VALUE``, the value of its ``.upit`` file), ``<name>.prec`` (its
precedences) and ``<name>_upit.sol`` (its published ultimate pit).

Reference: Espinoza, D., Goycoolea, M., Moreno, E., & Newman, A. (2013). MineLib: a library of open pit mining
problems. Annals of Operations Research, 206(1). https://doi.org/10.1007/s10479-012-1258-3
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import pymineflow

URL = "https://minelib.org/v1"
DATA = Path(__file__).parent / "data"

# the columns after "id x y z", as MineLib documents every instance (sm2 has none documented)
COLUMNS: dict[str, list[str] | None] = {
    "newman1": ["TYPE", "GRADE", "TONNS", "MIN_CAF", "VALUE_EXTRACC", "VALUE_PROC", "APRIORI_PROCESS"],
    "zuck_small": ["COST", "BLOCK_VALUE", "ROCK_TONNES", "ORE_TONNES"],
    "kd": ["TONN", "BLOCKVALUE", "DESTINATION", "CU_PCT", "PROCESS_PROFIT"],
    "zuck_medium": ["COST", "BLOCK_VALUE", "ROCK_TONNES", "ORE_TONNES"],
    "p4hd": ["TONN", "BLOCKVALUE", "DESTINATION", "AU_OZ_TON", "AG_OZ_TON", "CU_PCT"],
    "marvin": ["TONN", "AU_PPM", "CU_PCT", "PROC_PROFIT"],
    "w23": [
        "DEST",
        "PHASE",
        "AUREC",
        "AUFA",
        "TONS",
        "CO3",
        "ORGC",
        "SULF",
        "MCOST",
        "PCOST",
        "TCOST",
        "TVALUE",
        "BVALUE",
        "RC_STOCKPILE",
        "RC_ROCKCHAR",
    ],
    "zuck_large": ["COST", "BLOCK_VALUE", "ROCK_TONNES", "ORE_TONNES"],
    "sm2": None,
    "mclaughlin_limit": ["BLOCKVALUE", "TON", "DESTINATION", "AU_OZ_TON"],
}


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (PyMineFlow examples)"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return bytes(response.read())


def _download(name: str, folder: Path) -> None:
    for part in ("blocks", "prec", "files"):
        with zipfile.ZipFile(io.BytesIO(_get(f"{URL}/data/{name}.{part}.zip"))) as z:
            for member in z.namelist():
                if member.endswith((".blocks", ".prec", ".upit")):
                    (folder / Path(member).name).write_bytes(z.read(member))
    (folder / f"{name}_upit.sol").write_bytes(_get(f"{URL}/sols/{name}_upit.sol"))


def load(name: str, folder: Path = DATA) -> tuple[Path, Path, Path]:
    """The CSV, ``.prec`` and published solution of a MineLib UPIT instance (downloaded the first time)."""
    if name not in COLUMNS:
        raise ValueError(f"unknown MineLib instance {name!r}; use one of {list(COLUMNS)}")
    folder.mkdir(parents=True, exist_ok=True)
    csv, prec, sol = folder / f"{name}.csv", folder / f"{name}.prec", folder / f"{name}_upit.sol"
    if not (csv.exists() and prec.exists() and sol.exists()):
        needed = [folder / f"{name}.{part}" for part in ("blocks", "prec", "upit")] + [sol]
        if not all(p.exists() for p in needed):  # a first run, or one that was interrupted: download again
            print(f"downloading {name} from {URL} ...")
            _download(name, folder)
        blocks = pd.read_csv(folder / f"{name}.blocks", sep=r"\s+", header=None)
        extra = blocks.shape[1] - 4
        names = COLUMNS[name] or [f"ATTR_{k + 1}" for k in range(extra)]
        if len(names) != extra:
            raise ValueError(f"{name}.blocks has {extra} columns after id x y z, expected {len(names)}")
        blocks.columns = ["ID", "IX", "IY", "IZ", *names]
        if not np.array_equal(blocks["ID"].to_numpy(), np.arange(len(blocks))):
            raise ValueError(f"{name}.blocks: the ids must be 0, 1, 2, ... in order")
        blocks["VALUE"] = pymineflow.read_minelib_values(folder / f"{name}.upit")
        blocks.to_csv(csv, index=False)
    return csv, prec, sol
