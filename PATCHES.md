# Patches to MineFlow

PyMineFlow ships a copy of [MineFlow](https://github.com/MineFlowCSM/MineFlow) in `third_party/mineflow/` (MIT, © 2022 Matthew Deutsch). The copy is identical to the original except for the patches listed here. Each patch is marked in the code with a `[PyMineFlow]` comment.

MineFlow's license (`third_party/mineflow/LICENSE`) and its copyright notice are kept **unchanged**.

---

## 1. Slope interpolation with a negative azimuth (`SlopeDefinition::Get`)

**File:** `third_party/mineflow/mineflow.cpp`, function `SlopeDefinition::Get`. Six lines added; nothing changed or removed.

### The problem
`SlopeDefinition::Get(azimuth)` interpolates the slope between the two neighbouring azimuth–slope pairs:

- `GetLeftRight` normalizes the azimuth to `[0, 2π)`, but only **its own copy**, to choose the neighbours.
- `GetXval` receives the **non-normalized** azimuth and uses it for the interpolation weight.

`SlopeDefinition::Within(dx, dy, dz)` computes the azimuth as `π/2 − atan2(dy, dx)`, which is **negative** for directions with azimuth 270° to 360° (the north-west quadrant, including due west). With that negative value `GetXval` gets a wrong weight and the interpolated slope is wrong. As a result the precedence pattern (`MinSearch`, `Naive`, `LessNaive`) includes or excludes the wrong offsets.

### When it happens
All of these hold:

- **Non-constant slopes:** more than one azimuth–slope pair, with different slopes.
- **A direction with azimuth 270° to 360°.**
- **A pair at or after that azimuth** within that range. For example:
  - with pairs at 0/90/180/270, only due west is affected;
  - with a pair at 300° or 315°, the whole range from 270° up to that pair is affected.

**Not affected:** constant slopes (a single pair), directions from 0° to 270°, and the `1:5` / `1:9` patterns.

### Example
Slopes `{0°: 45°, 90°: 50°, 180°: 40°, 270°: 55°}`:

| Call | Original | Patched |
|---|---|---|
| `Get(270°)` | 55.0° | 55.0° |
| `Get(−90°)` (same direction) | **43.0°** | 55.0° |
| `Within(−1, 0, 1)` (west, 45° elevation) | **true** | false |

### Fix
At the start of `SlopeDefinition::Get`, after the empty and single-pair cases, the azimuth is normalized to `[0, 2π)`, so `GetLeftRight` and `GetXval` see the same value:

```cpp
// [PyMineFlow] ...
while (azimuth < 0) azimuth += TAU;
while (azimuth >= TAU) azimuth -= TAU;
```

`CubicInterpolate` and `CosineInterpolate` also call `GetXval`, but only with azimuths from `Linspace(0, TAU)`, which are never negative. They have no such problem and were not changed.

### Consequence
With non-constant slopes in the north-west quadrant, PyMineFlow's results **differ from the original MineFlow**, including its `--minsearch` executable. PyMineFlow's are the correct ones.

### Verification
- `tests/test_pit.py::test_azimuth_patch_regression_west_axis`: the due-west case with 0/90/180/270 sectors.
- `tests/test_pit.py::test_against_exact_cone_max_flow[sectors-*]`: the pit with sectors matches the maximum closure over the exact cone (scipy max-flow).

### Upstream status
To be reported at https://github.com/MineFlowCSM/MineFlow/issues.
