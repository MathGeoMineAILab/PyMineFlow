# MineLib: marvin

Mina de prueba Marvin; bloques de 30 x 30 x 30 m, precedencias: 45 grados, 8 niveles.

| Bloques | Arcos de precedencia | Óptimo UPIT publicado | PyMineFlow | Bloques en el pit |
|---:|---:|---:|---:|---:|
| 53,271 | 650,631 | 1,415,655,436 | 1,415,655,436 (el mismo pit, bloque a bloque) | 8,516 |

## Ejecutar

```bash
pip install pymineflow
python marvin.py
```

La primera vez se descarga la instancia de [MineLib](https://minelib.org) (CC BY-SA 3.0) en `../data/` y se
escribe como `marvin.csv`: el archivo `.blocks` de MineLib con los nombres de columna de su documentación
(`ID, IX, IY, IZ, ...`) y `VALUE`, el valor de cada bloque de su archivo `.upit`.

## Qué hace el código

1. **El pit final tal cual lo publica MineLib**, con sus propias precedencias (`marvin.prec`):
   `pymineflow.solve(csv, precedence=prec, value="VALUE")`. Imprime el reporte, compara el pit con la solución
   publicada por MineLib y escribe `results/marvin_pit.csv` (el modelo completo con `PIT` = 1 dentro),
   `results/marvin_pit_only.csv` y `results/marvin_pit_report.txt`.
2. **Un estudio completo** con el tamaño de bloque (30 x 30 x 30 m), 45° y 8 nivel(es) que describe MineLib:
   100 pits anidados (factores del 1 % al 100 % de los valores) con descuento top-down por bancos y después el
   OES del pit final, con la tasa de descuento y las capacidades del archivo `.cpit` de MineLib:

   | Tasa de descuento | Capacidad de mina | Capacidad de planta | Bancos por año* |
   |---:|---:|---:|---:|
   | 10 % | 60,000,000 t al año | 20,000,000 t al año | 1 |

   \* MineLib no lo da: son los bancos del pit final divididos por su vida. Mineral: los bloques de valor
   positivo.

   Escribe en `results/`: `report_cumulative.csv` y `report_incremental.csv` (cada fase con su VAN best y worst
   case y la secuencia), `model.csv` y `final_pit.csv`, `npv.png` (best contra worst case), `oes.gif` (el pit
   final minado en el orden de la secuencia) y `oes_surface.gif` (la superficie excavada), ambos en planta, y la
   secuencia bloque a bloque (`oes.csv`, `oes_periods.csv`, `oes_report.txt`, `oes_npv.png`). Después abre dos
   ventanas 3-D con una barra a lo largo de la secuencia: el pit final que se va minando y la superficie que se
   va excavando.

   Resultado: 63 fases, un pit final de 7,424 bloques, VAN del OES 804,377,504 en 13.39 períodos
   (cada bloque descontado cuando se mina). `oes.check()` confirma que cada bloque se mina una vez, después de
   sus predecesores y dentro de las capacidades.

