"""Block values from grades, prices and costs (optional): any number of metals and processes,
payable fractions, selling/refining costs, and contaminant limits and penalties.

For every block and every process ``p`` the economics are::

    revenue_p = sum over metals m of
                contained_m * recovery[m, p] * payable_m * (price_m * price_factor - selling_cost_m)
    value_p   = revenue_factor * revenue_p
                - sum over contaminants c of tonnage * penalty_c * max(0, grade_c - threshold_c)
                - tonnage * cost_p

A process is allowed only when every contaminant grade is within its limits (the contaminant's own
``limit`` and the process's ``limits``). The block goes to the best allowed process when its value
is positive, otherwise to waste, and its value is::

    value = max(0, best value_p) - tonnage * mining_cost

``contained_m`` is the metal in the block in the units of its price (e.g. ounces for $/oz), from
the grade, the grade unit, the tonnage and the tonnage unit. Prices and costs are per those units.

Every number (price, selling cost, payable, recovery, processing and mining cost, tonnage,
density, penalties, thresholds and limits) can be:

- a constant: ``price=1800``;
- a column of the table: ``mining_cost="MCOST"``;
- an array with one value per block: ``cost=np.array([...])``;
- a dict by domain, with ``Economics(domain="ROCK")``: ``recovery={"AU": {"oxide": 0.9, "sulphide": 0.8}}``.

A grade that is already recovered (``Metal(..., recovered=True)``) is never multiplied by a recovery
again. Increasing a price factor or the revenue factor never lowers a block's value, so the pits of
increasing factors are nested (see ``nested_pits``).
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["Contaminant", "Economics", "Metal", "Process", "economics_from_dict", "read_economics"]

_TROY_OUNCE_G = 31.1034768
_TONNAGE_UNIT_G = {"t": 1e6, "st": 907_184.74, "lt": 1_016_046.9088}  # metric, short and long ton
_PRICE_UNIT_G = {"g": 1.0, "kg": 1e3, "t": 1e6, "lb": 453.59237, "oz": _TROY_OUNCE_G}
_FRACTION = {"%": 1e-2, "ppm": 1e-6, "g/t": 1e-6, "ppb": 1e-9}  # mass fraction per grade unit
_TONNAGE_NAMES = ("TON", "TONS", "TONNES", "TONNAGE", "TONN", "TONNS", "TONELAJE")

Param = float | str | Sequence[float] | NDArray[Any] | Mapping[Any, float]
"""A number, a column name, one value per block, or a dict by domain (see the module documentation)."""

Value = float | NDArray[np.float64]


def _contained_per_grade(grade_unit: str, price_unit: str, tonnage_unit: str) -> float:
    """Metal contained (in price units) per grade unit and per tonnage unit of rock."""
    if tonnage_unit not in _TONNAGE_UNIT_G:
        raise ValueError(f"tonnage_unit must be one of {sorted(_TONNAGE_UNIT_G)}, got {tonnage_unit!r}")
    if price_unit not in _PRICE_UNIT_G:
        raise ValueError(f"price_unit must be one of {sorted(_PRICE_UNIT_G)}, got {price_unit!r}")
    rock_g = _TONNAGE_UNIT_G[tonnage_unit]
    if grade_unit == "oz/t":  # troy ounces per tonnage unit (oz per short ton with tonnage_unit="st")
        metal_g = _TROY_OUNCE_G
    elif grade_unit in _FRACTION:
        metal_g = _FRACTION[grade_unit] * rock_g
    else:
        raise ValueError(f"grade_unit must be one of {[*sorted(_FRACTION), 'oz/t']}, got {grade_unit!r}")
    return metal_g / _PRICE_UNIT_G[price_unit]


def _finite(name: str, x: float, low: float = -math.inf, high: float = math.inf) -> float:
    x = float(x)
    if not (math.isfinite(x) and low <= x <= high):
        raise ValueError(f"{name} must be a finite number in [{low}, {high}], got {x}")
    return x


def _check_array(name: str, a: NDArray[Any], low: float, high: float) -> NDArray[np.float64]:
    if a.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be numbers, got dtype {a.dtype}")
    a = a.astype(np.float64, copy=False)
    if a.size and (not np.isfinite(a).all() or a.min() < low or a.max() > high):
        bad = a[~(np.isfinite(a) & (a >= low) & (a <= high))][:3]
        raise ValueError(f"{name} must be finite and in [{low}, {high}] (no missing values), got e.g. {bad.tolist()}")
    return a


def _is_constant(spec: Any) -> bool:
    return isinstance(spec, numbers.Real) and not isinstance(spec, bool)


def _check_param(name: str, spec: Any, low: float = 0.0, high: float = math.inf) -> None:
    """Checks what can be checked before seeing the table (constants, dicts and arrays)."""
    if isinstance(spec, bool) or spec is None:
        raise TypeError(f"{name} must be a number, a column name, an array or a dict by domain, got {spec!r}")
    if _is_constant(spec):
        _finite(name, spec, low, high)
    elif isinstance(spec, str):
        if not spec:
            raise ValueError(f"{name}: empty column name")
    elif isinstance(spec, Mapping):
        if not spec:
            raise ValueError(f"{name}: empty dict by domain")
        for label, x in spec.items():
            if not _is_constant(x):
                raise TypeError(f"{name}[{label!r}] must be a number, got {x!r}")
            _finite(f"{name}[{label!r}]", x, low, high)
    else:
        a = np.asarray(spec)
        if a.ndim != 1:
            raise ValueError(f"{name} must be a number, a column name, a 1-D array or a dict, got shape {a.shape}")
        _check_array(name, a, low, high)


@dataclass(frozen=True)
class Metal:
    """A paid metal.

    Parameters
    ----------
    grade : column with its grade.
    price : price per ``price_unit`` (e.g. 1800 for $/oz).
    grade_unit : ``"%"``, ``"g/t"``, ``"ppm"``, ``"ppb"`` or ``"oz/t"`` (troy ounces per tonnage unit).
    price_unit : ``"oz"`` (troy), ``"g"``, ``"kg"``, ``"lb"`` or ``"t"``.
    payable : fraction of the recovered metal that is paid (smelter terms).
    selling_cost : refining, transport or selling cost per ``price_unit`` paid.
    recovered : ``True`` when the grade is already recovered (an effective grade): no process
        recovery is applied to it, and giving one other than 1 is an error.

    ``price``, ``payable`` and ``selling_cost`` may be numbers, columns, arrays or dicts by domain.
    """

    grade: str
    price: Param
    grade_unit: str = "g/t"
    price_unit: str = "oz"
    payable: Param = 1.0
    selling_cost: Param = 0.0
    recovered: bool = False

    def __post_init__(self) -> None:
        _check_param(f"{self.grade}: price", self.price, 0.0)
        _check_param(f"{self.grade}: payable", self.payable, 0.0, 1.0)
        _check_param(f"{self.grade}: selling_cost", self.selling_cost, 0.0)
        _contained_per_grade(self.grade_unit, self.price_unit, "t")  # validates the units


@dataclass(frozen=True)
class Process:
    """A process (plant, leach pad...).

    Parameters
    ----------
    name : label of the destination.
    cost : processing cost per tonnage unit (added to the mining cost of the block).
    recovery : fraction recovered of each metal, by grade column (metals not listed are not recovered).
    limits : maximum contaminant grade accepted by this process, by grade column.

    ``cost``, each recovery and each limit may be numbers, columns, arrays or dicts by domain.
    """

    name: str
    cost: Param
    recovery: Mapping[str, Param]
    limits: Mapping[str, Param] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_param(f"process {self.name}: cost", self.cost, 0.0)
        for metal, r in self.recovery.items():
            _check_param(f"process {self.name}: recovery of {metal}", r, 0.0, 1.0)
        for name, limit in self.limits.items():
            _check_param(f"process {self.name}: limit of {name}", limit, 0.0)


@dataclass(frozen=True)
class Contaminant:
    """A penalised or limited element (e.g. arsenic).

    Parameters
    ----------
    grade : column with its grade.
    penalty : cost per tonnage unit processed and per grade unit above ``threshold``.
    threshold : grade below which there is no penalty.
    limit : maximum grade that any process accepts (``None`` = no limit); above it the block is waste.
    """

    grade: str
    penalty: Param = 0.0
    threshold: Param = 0.0
    limit: Param | None = None

    def __post_init__(self) -> None:
        _check_param(f"{self.grade}: penalty", self.penalty, 0.0)
        _check_param(f"{self.grade}: threshold", self.threshold, 0.0)
        if self.limit is not None:
            _check_param(f"{self.grade}: limit", self.limit, 0.0)


def _table_rows(table: Mapping[str, Any]) -> int:
    for column in table.values():
        return int(np.asarray(column).shape[0])
    return 0


class _Resolver:
    """Turns parameters into numbers or per-row arrays for a table (and an optional row subset)."""

    def __init__(self, eco: Economics, table: Mapping[str, Any], rows: NDArray[np.intp] | None) -> None:
        self.table = table
        self.rows = None if rows is None else np.asarray(rows)
        self.n_total = _table_rows(table)
        self.n = self.n_total if self.rows is None else int(self.rows.size)
        self._domain: tuple[NDArray[Any], NDArray[np.intp]] | None = None
        self._domain_spec = eco.domain

    def take(self, a: NDArray[Any]) -> NDArray[Any]:
        return a if self.rows is None else a[self.rows]

    def column(self, name: str, what: str) -> NDArray[Any]:
        if name not in self.table:
            raise ValueError(f"{what} column {name!r} not found; available: {list(self.table)}")
        return self.take(np.asarray(self.table[name]))

    def domain(self) -> tuple[NDArray[Any], NDArray[np.intp]]:
        """(distinct labels, index of each row's label)."""
        if self._domain is None:
            spec = self._domain_spec
            if spec is None:
                raise ValueError("a parameter is given by domain (a dict): pass Economics(domain='COLUMN')")
            if isinstance(spec, str):
                labels = self.column(spec, "domain")
            else:
                labels = np.asarray(spec)
                if labels.shape != (self.n_total,):
                    raise ValueError(f"domain must hold one label per block ({self.n_total}), got shape {labels.shape}")
                labels = self.take(labels)
            unique, inverse = np.unique(labels, return_inverse=True)
            self._domain = unique, inverse.ravel()
        return self._domain

    def value(self, name: str, spec: Any, low: float = 0.0, high: float = math.inf) -> Value:
        """A number (constants stay numbers: the arithmetic is that of plain floats) or one per row."""
        if _is_constant(spec):
            return float(spec)
        if isinstance(spec, str):
            return _check_array(f"{name} (column {spec!r})", self.column(spec, name), low, high)
        if isinstance(spec, Mapping):
            labels, inverse = self.domain()
            keys = [_domain_key(spec, label) for label in labels.tolist()]
            missing = [label for label, key in zip(labels.tolist(), keys, strict=True) if key is None]
            if missing:
                raise ValueError(f"{name}: no value for the domains {missing[:10]}")
            per_label = np.array([float(spec[key]) for key in keys], dtype=np.float64)
            return _check_array(name, per_label, low, high)[inverse]
        a = np.asarray(spec)
        if a.shape != (self.n_total,):
            raise ValueError(f"{name} must hold one value per block ({self.n_total}), got shape {a.shape}")
        return _check_array(name, self.take(a), low, high)


def _domain_key(spec: Mapping[Any, Any], label: Any) -> Any:
    """The key of ``spec`` for a domain label: the label itself or, as JSON keys are text, its text (1 -> "1")."""
    if label in spec:
        return label
    if isinstance(label, float) and label.is_integer():
        label = int(label)
    text = str(label)
    return text if text in spec else None


@dataclass(frozen=True)
class _Terms:
    """The economics of every block split in parts (see ``Economics.terms``)."""

    revenue: list[NDArray[np.float64]]  # per process: revenue per tonnage unit, net of selling costs
    cost: list[Value]  # per process: processing cost per tonnage unit
    allowed: list[NDArray[np.bool_]]  # per process: contaminants within limits
    penalty: Value  # contaminant penalty per tonnage unit processed
    tons: NDArray[np.float64]
    mining: NDArray[np.float64]  # mining cost of the block (tonnage * mining cost)

    def margins(self, revenue_factor: float = 1.0) -> list[NDArray[np.float64]]:
        """Per process: the value of processing the block, before its mining cost (allowed or not)."""
        return [
            self.tons * (revenue_factor * r - self.penalty - c) for r, c in zip(self.revenue, self.cost, strict=True)
        ]


@dataclass(frozen=True)
class Economics:
    """Block economics (see the module documentation for the formula).

    Parameters
    ----------
    metals : the paid metals.
    processes : the possible processes (at least one).
    mining_cost : cost per tonnage unit of every mined block (ore or waste).
    tonnage : column with the tonnage of each block, a number, an array or a dict by domain. By default
        (``None``) the column ``TON``, or the model's one ``TONNAGE``, ``TONNES``, ... column.
    tonnage_unit : ``"t"`` (metric tonne), ``"st"`` (short ton) or ``"lt"`` (long ton).
    contaminants : penalised or limited elements.
    domain : column (or one label per block) for the parameters given by domain as dicts.
    density, block_size : instead of a tonnage, the density (a column, a number, an array or a dict
        by domain) and the block size (one number or ``(dx, dy, dz)``): ``tonnage = density * dx * dy * dz``.

    Every cost, price, recovery and the tonnage may be a number, a column, an array or a dict by
    domain (see the module documentation).

    Examples
    --------
    >>> eco = Economics(metals=[Metal("AU", price=1800, grade_unit="g/t", price_unit="oz")],
    ...                 processes=[Process("mill", cost=20, recovery={"AU": 0.9})],
    ...                 mining_cost=2, tonnage="TON")
    >>> table = {"AU": np.array([0.0, 1.0]), "TON": np.array([100.0, 100.0])}
    >>> values, destination = eco.block_values(table)
    >>> np.round(values, 2).tolist(), destination.tolist()
    ([-200.0, 3008.42], [-1, 0])
    """

    metals: Sequence[Metal]
    processes: Sequence[Process]
    mining_cost: Param
    tonnage: Param | None = None
    tonnage_unit: str = "t"
    contaminants: Sequence[Contaminant] = ()
    domain: str | Sequence[Any] | NDArray[Any] | None = None
    density: Param | None = None
    block_size: float | Sequence[float] | None = None

    def __post_init__(self) -> None:
        if not self.processes:
            raise ValueError("economics needs at least one process")
        _check_param("mining_cost", self.mining_cost, 0.0)
        if self.tonnage is not None:
            _check_param("tonnage", self.tonnage, 0.0)
        names = [m.grade for m in self.metals]
        if len(set(names)) != len(names):
            raise ValueError(f"repeated metal: {names}")
        recovered = {m.grade for m in self.metals if m.recovered}
        for p in self.processes:
            unknown = set(p.recovery) - set(names)
            if unknown:
                raise ValueError(f"process {p.name}: recovery of unknown metals {sorted(unknown)}")
            unknown = set(p.limits) - {c.grade for c in self.contaminants} - set(names)
            if unknown:
                raise ValueError(f"process {p.name}: limits on unknown columns {sorted(unknown)}")
            twice = sorted(
                m for m in recovered if m in p.recovery and not (_is_constant(p.recovery[m]) and p.recovery[m] == 1)
            )
            if twice:
                raise ValueError(
                    f"process {p.name}: {twice} are already recovered grades (recovered=True); "
                    "remove their recovery so that it is not applied twice"
                )
        for m in self.metals:
            _contained_per_grade(m.grade_unit, m.price_unit, self.tonnage_unit)
        if (self.density is None) != (self.block_size is None):
            raise ValueError("density= and block_size= go together (tonnage = density * block volume)")
        if self.density is not None:
            _check_param("density", self.density, 0.0)
            if self.tonnage is not None:
                raise ValueError("pass tonnage= or density= (with block_size=), not both")
            self._volume()

    def _volume(self) -> float:
        size = np.asarray(self.block_size, dtype=np.float64).ravel()
        if size.size == 1:
            size = np.repeat(size, 3)
        if size.size != 3 or not np.isfinite(size).all() or (size <= 0).any():
            raise ValueError(f"block_size must be one or three positive numbers, got {self.block_size!r}")
        return float(np.prod(size))

    @property
    def columns(self) -> list[str]:
        """Columns the economics read (grades, and the parameters given as column names)."""
        cols: list[Any] = [m.grade for m in self.metals] + [c.grade for c in self.contaminants]
        params: list[Any] = [self.mining_cost, self.density, self.domain]
        params.append(("TON" if self.tonnage is None else self.tonnage) if self.density is None else None)
        for m in self.metals:
            params += [m.price, m.payable, m.selling_cost]
        for p in self.processes:
            params += [p.cost, *p.recovery.values(), *p.limits.values()]
        for c in self.contaminants:
            params += [c.penalty, c.threshold, c.limit]
        cols += [s for s in params if isinstance(s, str)]
        return list(dict.fromkeys(cols))

    def _tonnage_column(self, table: Mapping[str, Any]) -> str:
        """The tonnage column given, or by default ``"TON"``, or the one tonnage column of the model."""
        if self.tonnage is not None:
            return str(self.tonnage)
        if "TON" in table:
            return "TON"
        upper = {str(c).strip().upper(): c for c in table}
        found = [upper[n] for n in _TONNAGE_NAMES if n in upper]
        if len(found) > 1:
            raise ValueError(f"several tonnage columns {found}: pass tonnage= (economics)")
        if not found:
            raise ValueError(
                f"no tonnage column (TON, TONNES, ...): pass tonnage= or density=; available: {list(table)}"
            )
        return str(found[0])

    def tonnages(self, table: Mapping[str, Any], rows: NDArray[np.intp] | None = None) -> NDArray[np.float64]:
        """The tonnage of every block (or of ``rows``) used by these economics."""
        return self._tons(_Resolver(self, table, rows))

    def _tons(self, r: _Resolver) -> NDArray[np.float64]:
        if self.density is not None:
            tons = np.asarray(r.value("density", self.density), dtype=np.float64) * self._volume()
        elif self.tonnage is None or isinstance(self.tonnage, str):
            name = self._tonnage_column(r.table)
            tons = np.asarray(r.column(name, "economics"), dtype=np.float64)
            if not np.isfinite(tons).all() or (tons < 0).any():
                raise ValueError(f"tonnage column {name!r} must be finite and >= 0")
        else:
            tons = np.asarray(r.value("tonnage", self.tonnage), dtype=np.float64)
        return np.broadcast_to(tons, (r.n,)).astype(np.float64, copy=True)

    def terms(
        self,
        table: Mapping[str, Any],
        price_factor: float = 1.0,
        *,
        rows: NDArray[np.intp] | None = None,
        grades: Mapping[str, Any] | None = None,
    ) -> _Terms:
        """The economics of every block split in parts: per process the revenue per tonnage unit (net
        of selling costs, at ``price_factor``), the processing cost and whether it is allowed; per block
        the contaminant penalty, the tonnage and the mining cost.

        ``rows`` restricts everything to those rows; ``grades`` replaces grade columns by arrays (one
        value per row returned), e.g. another estimate of the grades. The revenue is linear in the price
        factor, so ``terms(t, 1)`` and ``terms(t, 0)`` give the gross revenue and the selling costs."""
        _finite("price_factor", price_factor, 0.0)
        r = _Resolver(self, table, rows)
        tons = self._tons(r)
        grade_of: dict[str, NDArray[np.float64]] = {}
        for name in dict.fromkeys([m.grade for m in self.metals] + [c.grade for c in self.contaminants]):
            if grades is not None and name in grades:
                g = np.asarray(grades[name], dtype=np.float64)
                if g.shape != (r.n,):
                    raise ValueError(f"grades[{name!r}] must hold one value per block ({r.n}), got shape {g.shape}")
            else:
                if name not in table:
                    raise ValueError(f"economics columns [{name!r}] not found; available: {list(table)}")
                g = np.asarray(r.column(name, "grade"), dtype=np.float64)
            if not np.isfinite(g).all() or (g < 0).any():
                raise ValueError(f"grade column {name!r} must be finite and >= 0 (no missing values)")
            grade_of[name] = g
        penalty: Value = 0.0
        common = np.ones(r.n, dtype=bool)
        for c in self.contaminants:
            excess = grade_of[c.grade] - r.value(f"{c.grade}: threshold", c.threshold)
            penalty = penalty + r.value(f"{c.grade}: penalty", c.penalty) * np.maximum(excess, 0.0)
            if c.limit is not None:
                common &= grade_of[c.grade] <= r.value(f"{c.grade}: limit", c.limit)
        revenues: list[NDArray[np.float64]] = []
        costs: list[Value] = []
        allowed: list[NDArray[np.bool_]] = []
        for p in self.processes:
            revenue = np.zeros(r.n)
            for m in self.metals:
                if m.recovered:
                    rec: Value = 1.0
                else:
                    rec = r.value(f"process {p.name}: recovery of {m.grade}", p.recovery.get(m.grade, 0.0), 0.0, 1.0)
                if _is_constant(rec) and rec == 0.0:
                    continue
                price = r.value(f"{m.grade}: price", m.price)
                payable = r.value(f"{m.grade}: payable", m.payable, 0.0, 1.0)
                selling = r.value(f"{m.grade}: selling_cost", m.selling_cost)
                per_unit = rec * payable * (price * price_factor - selling)
                contained = _contained_per_grade(m.grade_unit, m.price_unit, self.tonnage_unit)
                revenue += grade_of[m.grade] * contained * per_unit
            ok = common.copy()
            for name, limit in p.limits.items():
                ok &= grade_of[name] <= r.value(f"process {p.name}: limit of {name}", limit)
            revenues.append(revenue)
            costs.append(r.value(f"process {p.name}: cost", p.cost))
            allowed.append(ok)
        mining = np.broadcast_to(tons * r.value("mining_cost", self.mining_cost), (r.n,)).astype(np.float64)
        return _Terms(revenues, costs, allowed, penalty, tons, mining)

    def block_values(
        self,
        table: Mapping[str, Any],
        price_factor: float = 1.0,
        revenue_factor: float = 1.0,
        *,
        rows: NDArray[np.intp] | None = None,
        grades: Mapping[str, Any] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.int16]]:
        """Value of every block and its destination (index into ``processes``, -1 = waste).

        ``price_factor`` multiplies every metal price; ``revenue_factor`` multiplies the revenue.
        ``rows`` restricts the result to those rows; ``grades`` replaces grade columns by arrays (one
        value per row returned), e.g. another estimate of the grades.
        """
        _finite("revenue_factor", revenue_factor, 0.0)
        t = self.terms(table, price_factor, rows=rows, grades=grades)
        best = np.zeros_like(t.tons)  # waste: 0 beyond the mining cost
        destination = np.full(t.tons.shape, -1, dtype=np.int16)
        for k, (value, ok) in enumerate(zip(t.margins(revenue_factor), t.allowed, strict=True)):
            better = ok & (value > best)
            best[better] = value[better]
            destination[better] = k
        return best - t.mining, destination

    def destination_names(self) -> list[str]:
        """Names by destination index; waste (-1) is the last one."""
        return [p.name for p in self.processes] + ["waste"]


def economics_from_dict(data: Mapping[str, Any]) -> Economics:
    """Economics from plain data (e.g. a JSON file), with the same names as the classes.

    Examples
    --------
    >>> eco = economics_from_dict({
    ...     "metals": [{"grade": "AU", "price": 900, "grade_unit": "oz/t", "price_unit": "oz"}],
    ...     "processes": [{"name": "mill", "cost": 19, "recovery": {"AU": 0.9}}],
    ...     "mining_cost": 1.32, "tonnage": "TON", "tonnage_unit": "st"})
    >>> eco.processes[0].name, eco.metals[0].price
    ('mill', 900)
    """
    known = {"metals", "processes", "mining_cost", "tonnage", "tonnage_unit", "contaminants", "domain",
             "density", "block_size"}  # fmt: skip
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown economics keys {sorted(unknown)}; use {sorted(known)}")
    if "mining_cost" not in data:
        raise ValueError("the economics need a mining_cost (cost per tonnage unit of every block mined)")
    return Economics(
        metals=[Metal(**m) for m in data.get("metals", [])],
        processes=[Process(**p) for p in data.get("processes", [])],
        mining_cost=data["mining_cost"],
        tonnage=data.get("tonnage"),
        tonnage_unit=data.get("tonnage_unit", "t"),
        contaminants=[Contaminant(**c) for c in data.get("contaminants", [])],
        domain=data.get("domain"),
        density=data.get("density"),
        block_size=data.get("block_size"),
    )


def read_economics(path: str | Path) -> Economics:
    """Economics from a JSON file (see ``economics_from_dict``)."""
    import json

    with Path(path).open(encoding="utf-8") as f:
        return economics_from_dict(json.load(f))
