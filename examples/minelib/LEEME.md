# Ejemplos de MineLib

[Read in English](README.md)

Todas las instancias a cielo abierto de [MineLib](https://minelib.org) (Espinoza, Goycoolea, Moreno y Newman,
2013), una carpeta cada una, con un código, una explicación en inglés (`README.md`) y en español (`LEEME.md`).
Los datos se descargan de minelib.org la primera vez (`minelib_data.py`, en `data/`; MineLib tiene licencia
CC BY-SA 3.0) y se escriben como CSV con los nombres de columna de la documentación de MineLib.

Cada código resuelve el pit final con las precedencias de MineLib y lo compara con la solución publicada; cuando
MineLib documenta el tamaño de bloque y el talud, hace además un estudio completo (pits anidados con descuento
top-down, OES del pit final, reportes acumulado e incremental, gráfico best contra worst case y los GIF de la
secuencia y de la superficie) con la tasa de descuento y las capacidades de MineLib.

| Instancia | Bloques | Óptimo UPIT publicado | Bloques en el pit | Estudio completo |
|---|---:|---:|---:|---|
| [newman1](newman1/) | 1,060 | 26,086,899 | 1,059 | sí |
| [zuck_small](zuck_small/) | 9,400 | 1,422,726,898 | 9,399 | solo UPIT (sin geometría) |
| [kd](kd/) | 14,153 | 652,195,037 | 12,154 | sí |
| [zuck_medium](zuck_medium/) | 29,277 | 1,075,124,490 | 27,387 | solo UPIT (sin geometría) |
| [p4hd](p4hd/) | 40,947 | 293,373,256 | 28,844 | sí |
| [marvin](marvin/) | 53,271 | 1,415,655,436 | 8,516 | sí |
| [w23](w23/) | 74,260 | 510,973,998 | 49,061 | sí |
| [zuck_large](zuck_large/) | 96,821 | 122,220,280 | 96,821 | solo UPIT (sin geometría) |
| [sm2](sm2/) | 99,014 | 2,743,603,730 | 18,388 | solo UPIT (sin geometría) |
| [mclaughlin_limit](mclaughlin_limit/) | 112,687 | 1,495,726,474 | 110,225 | sí |

Cada pit de PyMineFlow es igual al publicado, bloque a bloque. Ejecute uno con `python kd/kd.py`, o todos:

```bash
for n in newman1 zuck_small kd zuck_medium p4hd marvin w23 zuck_large sm2 mclaughlin_limit; do python $n/$n.py; done
```

Al final los códigos abren ventanas 3-D: cierre cada una para continuar.
