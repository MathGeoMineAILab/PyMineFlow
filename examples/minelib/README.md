# MineLib examples

[Leer en español](LEEME.md)

Every open-pit instance of [MineLib](https://minelib.org) (Espinoza, Goycoolea, Moreno and Newman, 2013), one
folder each, with a script, an explanation in English (`README.md`) and in Spanish (`LEEME.md`). The data are
downloaded from minelib.org the first time (`minelib_data.py`, into `data/`; MineLib is licensed under CC BY-SA
3.0) and written as CSV files with the column names of MineLib's documentation.

Every script solves the ultimate pit with MineLib's own precedences and checks it against the published solution;
when MineLib documents the block size and slope, it also runs a full study (nested pits with top-down discounting,
the ultimate pit OES, the cumulative and incremental reports, the best against worst case chart, and the GIFs of
the sequence and of the ground surface) with MineLib's discount rate and capacities.

| Instance | Blocks | Published UPIT optimum | Blocks in the pit | Full study |
|---|---:|---:|---:|---|
| [newman1](newman1/) | 1,060 | 26,086,899 | 1,059 | yes |
| [zuck_small](zuck_small/) | 9,400 | 1,422,726,898 | 9,399 | UPIT only (no geometry) |
| [kd](kd/) | 14,153 | 652,195,037 | 12,154 | yes |
| [zuck_medium](zuck_medium/) | 29,277 | 1,075,124,490 | 27,387 | UPIT only (no geometry) |
| [p4hd](p4hd/) | 40,947 | 293,373,256 | 28,844 | yes |
| [marvin](marvin/) | 53,271 | 1,415,655,436 | 8,516 | yes |
| [w23](w23/) | 74,260 | 510,973,998 | 49,061 | yes |
| [zuck_large](zuck_large/) | 96,821 | 122,220,280 | 96,821 | UPIT only (no geometry) |
| [sm2](sm2/) | 99,014 | 2,743,603,730 | 18,388 | UPIT only (no geometry) |
| [mclaughlin_limit](mclaughlin_limit/) | 112,687 | 1,495,726,474 | 110,225 | yes |

Every PyMineFlow pit equals the published one, block for block. Run one with `python kd/kd.py`, or all of them:

```bash
for n in newman1 zuck_small kd zuck_medium p4hd marvin w23 zuck_large sm2 mclaughlin_limit; do python $n/$n.py; done
```

The scripts open 3-D windows at the end: close each one to go on.
