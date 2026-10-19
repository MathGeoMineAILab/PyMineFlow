# MineLib: zuck_small

Mina entregada por Mark Zuckerberg; bloques de tamaño desconocido, precedencias calculadas por quien la entregó.

| Bloques | Arcos de precedencia | Óptimo UPIT publicado | PyMineFlow | Bloques en el pit |
|---:|---:|---:|---:|---:|
| 9,400 | 145,640 | 1,422,726,898 | 1,422,726,898 (el mismo pit, bloque a bloque) | 9,399 |

## Ejecutar

```bash
pip install pymineflow
python zuck_small.py
```

La primera vez se descarga la instancia de [MineLib](https://minelib.org) (CC BY-SA 3.0) en `../data/` y se
escribe como `zuck_small.csv`: el archivo `.blocks` de MineLib con los nombres de columna de su documentación
(`ID, IX, IY, IZ, ...`) y `VALUE`, el valor de cada bloque de su archivo `.upit`.

## Qué hace el código

1. **El pit final tal cual lo publica MineLib**, con sus propias precedencias (`zuck_small.prec`):
   `pymineflow.solve(csv, precedence=prec, value="VALUE")`. Imprime el reporte, compara el pit con la solución
   publicada por MineLib y escribe `results/zuck_small_pit.csv` (el modelo completo con `PIT` = 1 dentro),
   `results/zuck_small_pit_only.csv` y `results/zuck_small_pit_report.txt`.

MineLib no da el tamaño de bloque ni el talud de esta instancia (las precedencias las calculó quien entregó los
datos), así que los pits anidados y la secuencia, que construyen los taludes a partir de la geometría, no se
aplican. El código guarda además una vista en planta del pit final (`results/zuck_small_pit.png`) y abre una vista
3-D.

