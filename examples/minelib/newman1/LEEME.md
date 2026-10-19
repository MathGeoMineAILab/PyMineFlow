# MineLib: newman1

Mina entregada por Alexandra Newman; bloques de tamaño desconocido, precedencias: los 5 bloques de arriba.

| Bloques | Arcos de precedencia | Óptimo UPIT publicado | PyMineFlow | Bloques en el pit |
|---:|---:|---:|---:|---:|
| 1,060 | 3,922 | 26,086,899 | 26,086,899 (el mismo pit, bloque a bloque) | 1,059 |

## Ejecutar

```bash
pip install pymineflow
python newman1.py
```

La primera vez se descarga la instancia de [MineLib](https://minelib.org) (CC BY-SA 3.0) en `../data/` y se
escribe como `newman1.csv`: el archivo `.blocks` de MineLib con los nombres de columna de su documentación
(`ID, IX, IY, IZ, ...`) y `VALUE`, el valor de cada bloque de su archivo `.upit`.

## Qué hace el código

1. **El pit final tal cual lo publica MineLib**, con sus propias precedencias (`newman1.prec`):
   `pymineflow.solve(csv, precedence=prec, value="VALUE")`. Imprime el reporte, compara el pit con la solución
   publicada por MineLib y escribe `results/newman1_pit.csv` (el modelo completo con `PIT` = 1 dentro),
   `results/newman1_pit_only.csv` y `results/newman1_pit_report.txt`.
2. **Un estudio completo** con el tamaño de bloque (cubos unitarios (tamaño desconocido), un nivel: los 5 bloques de arriba), 45° y 1 nivel(es) que describe MineLib:
   100 pits anidados (factores del 1 % al 100 % de los valores) con descuento top-down por bancos y después el
   OES del pit final, con la tasa de descuento y las capacidades del archivo `.cpit` de MineLib:

   | Tasa de descuento | Capacidad de mina | Capacidad de planta | Bancos por año* |
   |---:|---:|---:|---:|
   | 8 % | 2,000,000 t al año | 1,100,000 t al año | 7 |

   \* MineLib no lo da: son los bancos del pit final divididos por su vida. Mineral: los bloques de valor
   positivo.

   Escribe en `results/`: `report_cumulative.csv` y `report_incremental.csv` (cada fase con su VAN best y worst
   case y la secuencia), `model.csv` y `final_pit.csv`, `npv.png` (best contra worst case), `oes.gif` (el pit
   final minado en el orden de la secuencia) y `oes_surface.gif` (la superficie excavada), ambos en planta, y la
   secuencia bloque a bloque (`oes.csv`, `oes_periods.csv`, `oes_report.txt`, `oes_npv.png`). Después abre dos
   ventanas 3-D con una barra a lo largo de la secuencia: el pit final que se va minando y la superficie que se
   va excavando.

   Resultado: 5 fases, un pit final de 1,060 bloques, VAN del OES 22,177,220 en 3.24 períodos
   (cada bloque descontado cuando se mina). `oes.check()` confirma que cada bloque se mina una vez, después de
   sus predecesores y dentro de las capacidades.

   Con taludes construidos desde la geometría el pit difiere un poco del de MineLib: MineLib quita los arcos hacia bloques que faltan en el modelo y PyMineFlow mantiene el cono a través de ellos.

