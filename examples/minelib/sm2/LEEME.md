# MineLib: sm2

Mina entregada por Alexandra Newman; bloques de tamaño desconocido, columnas no documentadas (ATTR_1 a ATTR_7).

| Bloques | Arcos de precedencia | Óptimo UPIT publicado | PyMineFlow | Bloques en el pit |
|---:|---:|---:|---:|---:|
| 99,014 | 96,642 | 2,743,603,730 | 2,743,603,730 (el mismo pit, bloque a bloque) | 18,388 |

## Ejecutar

```bash
pip install pymineflow
python sm2.py
```

La primera vez se descarga la instancia de [MineLib](https://minelib.org) (CC BY-SA 3.0) en `../data/` y se
escribe como `sm2.csv`: el archivo `.blocks` de MineLib con los nombres de columna de su documentación
(`ID, IX, IY, IZ, ...`) y `VALUE`, el valor de cada bloque de su archivo `.upit`.

## Qué hace el código

1. **El pit final tal cual lo publica MineLib**, con sus propias precedencias (`sm2.prec`):
   `pymineflow.solve(csv, precedence=prec, value="VALUE")`. Imprime el reporte, compara el pit con la solución
   publicada por MineLib y escribe `results/sm2_pit.csv` (el modelo completo con `PIT` = 1 dentro),
   `results/sm2_pit_only.csv` y `results/sm2_pit_report.txt`.

MineLib no da el tamaño de bloque ni el talud de esta instancia (las precedencias las calculó quien entregó los
datos), así que los pits anidados y la secuencia, que construyen los taludes a partir de la geometría, no se
aplican. El código guarda además una vista en planta del pit final (`results/sm2_pit.png`) y abre una vista
3-D.

