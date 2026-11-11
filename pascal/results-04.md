# Fork Pascal — séance 04 (liste de tokens pour la tête MTP, profil)

- Date : 2026-11-11 09:34 UTC
- GPU : Tesla P40, 580.126.09, 24576 MiB
- Base : aa75ff3239f9cc675b081697deda9e1b112345c6 + 01 + 02 + 04

## 1. Compilation

- compilé en 426 s

## 2. Profil par type de nœud (llama-bench, génération 1 token à la fois, sans CUDA graphs)

Temps mur entre deux synchronisations du flux : les très petits noyaux sont un peu surestimés. Les 30 premières lignes :
```
```

## 3. Liste de tokens pour la tête MTP

- corpus : 7 fichier(s), 5749431 caractères, 1656654 tokens, 17765 tokens distincts ; liste écrite : 18125 identifiants
- couverture des tokens du prompt de test (18854 tokens) par les N premiers de la liste :
  - N = 4096 : 69.40 %
  - N = 8192 : 75.29 %
  - N = 16384 : 82.27 %
  - N = 32768 : 87.21 %
  - N = 65536 : 87.21 %

## 4. Génération réelle (llama-server, q40, -fa on, -c 40000, 600 tokens, temp 0, MTP n-max 2 sauf mention)

| variante | prompt | tokens prompt | tokens générés | tg t/s | brouillons acceptés/proposés |
|---|---|---|---|---|---|
| référence p02 (rows par K), MTP2 | court | 2171 | 600 | **88.22** | 358/482 |
| référence p02 (rows par K), MTP2 | long | 18861 | 600 | **71.34** | 331/535 |
| 04 défaut (2 lignes), MTP2 | court | 2171 | 600 | **88.28** | 358/482 |
| 04 défaut (2 lignes), MTP2 | long | 18861 | 600 | **70.41** | 331/535 |
| 04 liste N=16384, MTP2 | court | 2171 | 600 | **91.56** | 342/512 |
| 04 liste N=16384, MTP2 | long | 18861 | 600 | **77.22** | 326/545 |
| 04 liste N=32768, MTP2 | court | 2171 | 600 | **91.44** | 347/502 |
| 04 liste N=32768, MTP2 | long | 18861 | 600 | **77.9** | 328/541 |
| 04 liste N=8192, MTP2 | court | 2171 | 600 | **81.85** | 308/579 |
| 04 liste N=8192, MTP2 | long | 18861 | 600 | **74.77** | 316/565 |
| 04 liste N=65536, MTP2 | court | 2171 | 600 | **91.53** | 347/502 |
| 04 liste N=65536, MTP2 | long | 18861 | 600 | **77.67** | 328/541 |
| référence p02 (2e passage), MTP2 | court | 2171 | 600 | **89.07** | 358/482 |
| référence p02 (2e passage), MTP2 | long | 18861 | 600 | **71.85** | 331/535 |
| 04 liste N=16384 (2e passage), MTP2 | court | 2171 | 600 | **89.39** | 342/512 |
| 04 liste N=16384 (2e passage), MTP2 | long | 18861 | 600 | **77.06** | 326/545 |
| 04 liste N=16384, MTP n-max 3 | court | 2171 | 600 | **86.21** | 380/656 |
| 04 liste N=16384, MTP n-max 3 | long | 18861 | 600 | **61.67** | 319/838 |
| 04 défaut, sans MTP | court | 2171 | 600 | **66.3** | —/— |
| 04 défaut, sans MTP | long | 18861 | 600 | **60.19** | —/— |

## 5. Profil d'un pas MTP n-max 2 (llama-server, prompt court, sans CUDA graphs)

Inclut le traitement du prompt (lignes à beaucoup de colonnes) ; lire surtout les lignes cols=1x1, 2x1, 3x1, 1x2, 1x3.
- tg sous profil (non comparable) : 2171 | 600 | **66.13** | 358/482
```
[ggml-cuda profile] 972 graphs, 9338.1 ms in nodes
[ggml-cuda profile]         ms      %     count   us/call  node
[ggml-cuda profile]    1160.42  12.43      9881     117.4  MUL_MAT_ID q4_0 2048x512x256 cols=1x3 +fused
[ggml-cuda profile]     700.18   7.50       485    1443.7  MUL_MAT q6_K 2048x248320x1 cols=1x1
[ggml-cuda profile]     490.44   5.25      7350      66.7  GATED_DELTA_NET +fused
[ggml-cuda profile]     486.78   5.21      8676      56.1  MUL_MAT_ID q4_0 512x2048x256 cols=8x3
[ggml-cuda profile]     469.08   5.02     51884       9.0  RMS_NORM +fused
[ggml-cuda profile]     431.83   4.62      8917      48.4  MUL_MAT q4_0 2048x8192x1 cols=3x1
[ggml-cuda profile]     428.35   4.59      3177     134.8  FLASH_ATTN_EXT
[ggml-cuda profile]     392.95   4.21       241    1630.5  MUL_MAT q6_K 2048x248320x1 cols=3x1
[ggml-cuda profile]     254.00   2.72      7230      35.1  MUL_MAT q4_0 2048x4096x1 cols=3x1
[ggml-cuda profile]     241.09   2.58     16154      14.9  GET_ROWS
[ggml-cuda profile]     230.47   2.47      6748      34.2  MUL_MAT q4_0 4096x2048x1 cols=3x1
[ggml-cuda profile]     229.92   2.46        82    2804.0  MUL_MAT_ID q4_0 2048x512x256 cols=1x2044
[ggml-cuda profile]     221.13   2.37      8077      27.4  CONCAT
[ggml-cuda profile]     197.41   2.11     14460      13.7  MUL_MAT q8_0 2048x512x1 cols=3x1
[ggml-cuda profile]     179.44   1.92      9640      18.6  MUL_MAT f32 2048x256x1 cols=3x1
[ggml-cuda profile]     172.75   1.85     10527      16.4  SOFT_MAX +fused
[ggml-cuda profile]     167.41   1.79     14460      11.6  MUL_MAT f32 2048x32x1 cols=3x1
[ggml-cuda profile]     163.69   1.75      3374      48.5  MUL_MAT q8_0 4096x2048x1 cols=3x1
[ggml-cuda profile]     163.55   1.75     25227       6.5  UNARY
[ggml-cuda profile]     162.94   1.74     17395       9.4  ADD
[ggml-cuda profile]     159.32   1.71     17877       8.9  MUL
[ggml-cuda profile]     143.81   1.54     10527      13.7  MUL +fused
[ggml-cuda profile]     135.62   1.45      9640      14.1  MUL_MAT q5_0 2048x512x1 cols=3x1
[ggml-cuda profile]     117.62   1.26        37    3178.9  MUL_MAT q4_0 2048x8192x1 cols=2044x1
[ggml-cuda profile]     105.38   1.13        36    2927.1  MUL_MAT_ID q4_0 512x2048x256 cols=8x2044
[ggml-cuda profile]     104.89   1.12      9640      10.9  MUL_MAT f32 2048x1x1 cols=3x1
[ggml-cuda profile]     100.62   1.08     10771       9.3  SET_ROWS
[ggml-cuda profile]      99.41   1.06     10527       9.4  ADD +fused
[ggml-cuda profile]      96.87   1.04      7350      13.2  SSM_CONV +fused
[ggml-cuda profile]      92.92   1.00     10527       8.8  UNARY +fused
[ggml-cuda profile]      85.30   0.91      9883       8.6  GLU
[ggml-cuda profile]      74.54   0.80      1205      61.9  MUL_MAT_ID q4_1 512x2048x256 cols=8x3
[ggml-cuda profile]      70.47   0.75      4820      14.6  MUL_MAT q8_0 512x2048x1 cols=3x1
[ggml-cuda profile]      70.42   0.75       964      73.0  MUL_MAT q6_K 2048x8192x1 cols=3x1
[ggml-cuda profile]      62.50   0.67      4820      13.0  MUL_MAT q5_0 512x2048x1 cols=3x1
[ggml-cuda profile]      52.61   0.56      7470       7.0  SCALE
[ggml-cuda profile]      50.60   0.54      3177      15.9  CONT
[ggml-cuda profile]      50.09   0.54        30    1669.6  MUL_MAT q4_0 2048x4096x1 cols=2044x1
[ggml-cuda profile]      48.77   0.52        28    1741.8  MUL_MAT q4_0 4096x2048x1 cols=2044x1
[ggml-cuda profile]      46.99   0.50      7350       6.4  CPY
[ggml-cuda profile]      43.02   0.46        82     524.6  MUL_MAT_ID q4_0 2048x512x256 cols=1x123
[ggml-cuda profile]      43.00   0.46      7350       5.9  ARANGE
[ggml-cuda profile]      34.74   0.37      3177      10.9  ROPE
```

_Terminé : 2026-11-11 09:51 UTC, durée 980 s. Journaux dans /root/fork4/logs._
# Séance 04 — suite

## A. Profil par type de nœud (llama-bench tg, 1 token à la fois, -fa on, sans CUDA graphs)

```
| qwen35moe 35B.A3B Q4_0         |  19.40 GiB |    35.51 B | CUDA       |  99 |   1 |            tg96 |         46.76 ± 0.00 |
[ggml-cuda profile] 97 graphs, 1995.1 ms in nodes
[ggml-cuda profile]         ms      %     count   us/call  node
[ggml-cuda profile]     190.69   9.56      3880      49.1  MUL_MAT_ID q4_0 2048x512x256 cols=1x1 +fused
[ggml-cuda profile]     163.66   8.20      3492      46.9  MUL_MAT q4_0 2048x8192x1 cols=1x1
[ggml-cuda profile]     154.90   7.76     18527       8.4  RMS_NORM +fused
[ggml-cuda profile]     139.84   7.01        97    1441.7  MUL_MAT q6_K 2048x248320x1 cols=1x1
[ggml-cuda profile]      95.35   4.78      2910      32.8  MUL_MAT q4_0 2048x4096x1 cols=1x1
[ggml-cuda profile]      93.33   4.68      3395      27.5  MUL_MAT_ID q4_0 512x2048x256 cols=8x1
[ggml-cuda profile]      88.78   4.45      6014      14.8  GET_ROWS
[ggml-cuda profile]      78.24   3.92      2425      32.3  MUL_MAT q4_0 4096x2048x1 cols=1x1 +fused
[ggml-cuda profile]      73.31   3.67      2910      25.2  GATED_DELTA_NET +fused
[ggml-cuda profile]      69.14   3.47      5820      11.9  MUL_MAT f32 2048x32x1 cols=1x1
[ggml-cuda profile]      62.80   3.15      3880      16.2  SOFT_MAX +fused
[ggml-cuda profile]      62.71   3.14      1358      46.2  MUL_MAT q8_0 4096x2048x1 cols=1x1 +fused
[ggml-cuda profile]      62.07   3.11      3880      16.0  MUL_MAT f32 2048x256x1 cols=1x1
[ggml-cuda profile]      56.13   2.81      6790       8.3  UNARY +fused
[ggml-cuda profile]      44.27   2.22      1940      22.8  MUL_MAT q8_0 2048x512x1 cols=1x1 +fused
[ggml-cuda profile]      43.56   2.18      1940      22.5  MUL_MAT q5_0 2048x512x1 cols=1x1 +fused
[ggml-cuda profile]      43.39   2.17      3880      11.2  MUL +fused
[ggml-cuda profile]      42.59   2.13      6790       6.3  UNARY
[ggml-cuda profile]      38.22   1.92      2910      13.1  CPY
[ggml-cuda profile]      35.25   1.77      3880       9.1  MUL_MAT f32 2048x1x1 cols=1x1
[ggml-cuda profile]      33.39   1.67      3880       8.6  MUL
[ggml-cuda profile]      33.20   1.66      3880       8.6  ADD +fused
[ggml-cuda profile]      32.82   1.64      1940      16.9  MUL_MAT q8_0 512x2048x1 cols=1x1
[ggml-cuda profile]      32.76   1.64       970      33.8  FLASH_ATTN_EXT
[ggml-cuda profile]      31.83   1.60      3007      10.6  ADD
[ggml-cuda profile]      30.24   1.52      1940      15.6  MUL_MAT q5_0 512x2048x1 cols=1x1
[ggml-cuda profile]      28.90   1.45      2910       9.9  CONCAT
[ggml-cuda profile]      26.48   1.33      2910       9.1  SSM_CONV +fused
[ggml-cuda profile]      24.07   1.21       388      62.0  MUL_MAT q6_K 2048x8192x1 cols=1x1
[ggml-cuda profile]      23.39   1.17      1940      12.1  MUL_MAT q8_0 2048x512x1 cols=1x1
[ggml-cuda profile]      15.12   0.76       485      31.2  MUL_MAT_ID q4_1 512x2048x256 cols=8x1
[ggml-cuda profile]      11.96   0.60       970      12.3  CONT
[ggml-cuda profile]      10.70   0.54       970      11.0  ROPE
[ggml-cuda profile]       8.87   0.44       970       9.1  ROPE +fused
[ggml-cuda profile]       8.47   0.42       970       8.7  SET_ROWS
[ggml-cuda profile]       3.03   0.15        97      31.3  MUL_MAT q4_0 4096x2048x1 cols=1x1
[ggml-cuda profile]       1.67   0.08       120      13.9  SCALE
```

## B. Listes de tokens mieux couvrantes

- liste 2 = romans (apostrophes droites et courbes) + seconde moitié du prompt : 18272 identifiants ; couvre 97.28 % du prompt court (hors échantillon) et 91.82 % du prompt entier
- liste 3 = liste 2 + prompt entier (cas « liste construite sur ses propres textes », dans l'échantillon) : 18783 identifiants

| variante | prompt | tokens prompt | tokens générés | tg t/s | brouillons acceptés/proposés |
|---|---|---|---|---|---|
| 04 liste 2, MTP2 | court | 2171 | 600 | **95.01** | 351/494 |
| 04 liste 2, MTP2 | long | 18861 | 600 | **77.8** | 329/539 |
| 04 liste 3, MTP2 | court | 2171 | 600 | **95.14** | 352/492 |
| 04 liste 3, MTP2 | long | 18861 | 600 | **77.49** | 329/539 |

## C. Profil d'un pas MTP n-max 2 (serveur, prompt court, sans CUDA graphs ; inclut le traitement du prompt)

- tg sous profil (non comparable) : 2171 | 600 | **63.52** | 358/482
```
[ggml-cuda profile] 972 graphs, 9482.7 ms in nodes
[ggml-cuda profile]         ms      %     count   us/call  node
[ggml-cuda profile]    1163.93  12.27      9881     117.8  MUL_MAT_ID q4_0 2048x512x256 cols=1x3 +fused
[ggml-cuda profile]     700.49   7.39       485    1444.3  MUL_MAT q6_K 2048x248320x1 cols=1x1
[ggml-cuda profile]     493.18   5.20      7350      67.1  GATED_DELTA_NET +fused
[ggml-cuda profile]     491.12   5.18      8676      56.6  MUL_MAT_ID q4_0 512x2048x256 cols=8x3
[ggml-cuda profile]     490.25   5.17     51884       9.4  RMS_NORM +fused
[ggml-cuda profile]     435.50   4.59      8917      48.8  MUL_MAT q4_0 2048x8192x1 cols=3x1
[ggml-cuda profile]     429.66   4.53      3177     135.2  FLASH_ATTN_EXT
[ggml-cuda profile]     392.87   4.14       241    1630.2  MUL_MAT q6_K 2048x248320x1 cols=3x1
[ggml-cuda profile]     257.39   2.71      7230      35.6  MUL_MAT q4_0 2048x4096x1 cols=3x1
[ggml-cuda profile]     245.47   2.59     16154      15.2  GET_ROWS
[ggml-cuda profile]     234.68   2.47      6748      34.8  MUL_MAT q4_0 4096x2048x1 cols=3x1
[ggml-cuda profile]     229.99   2.43        82    2804.8  MUL_MAT_ID q4_0 2048x512x256 cols=1x2044
[ggml-cuda profile]     223.17   2.35      8077      27.6  CONCAT
[ggml-cuda profile]     202.30   2.13     14460      14.0  MUL_MAT q8_0 2048x512x1 cols=3x1
[ggml-cuda profile]     182.28   1.92      9640      18.9  MUL_MAT f32 2048x256x1 cols=3x1
[ggml-cuda profile]     180.62   1.90     10527      17.2  SOFT_MAX +fused
[ggml-cuda profile]     173.14   1.83     14460      12.0  MUL_MAT f32 2048x32x1 cols=3x1
[ggml-cuda profile]     170.69   1.80     25227       6.8  UNARY
[ggml-cuda profile]     167.80   1.77     17395       9.6  ADD
[ggml-cuda profile]     166.98   1.76      3374      49.5  MUL_MAT q8_0 4096x2048x1 cols=3x1
[ggml-cuda profile]     162.89   1.72     17877       9.1  MUL
[ggml-cuda profile]     153.29   1.62     10527      14.6  MUL +fused
[ggml-cuda profile]     140.40   1.48      9640      14.6  MUL_MAT q5_0 2048x512x1 cols=3x1
[ggml-cuda profile]     117.51   1.24        37    3175.9  MUL_MAT q4_0 2048x8192x1 cols=2044x1
[ggml-cuda profile]     107.05   1.13      9640      11.1  MUL_MAT f32 2048x1x1 cols=3x1
[ggml-cuda profile]     105.38   1.11        36    2927.2  MUL_MAT_ID q4_0 512x2048x256 cols=8x2044
[ggml-cuda profile]     104.02   1.10     10771       9.7  SET_ROWS
[ggml-cuda profile]     102.45   1.08     10527       9.7  ADD +fused
[ggml-cuda profile]     100.14   1.06      7350      13.6  SSM_CONV +fused
[ggml-cuda profile]      96.79   1.02     10527       9.2  UNARY +fused
[ggml-cuda profile]      87.47   0.92      9883       8.9  GLU
[ggml-cuda profile]      75.50   0.80      1205      62.7  MUL_MAT_ID q4_1 512x2048x256 cols=8x3
[ggml-cuda profile]      72.38   0.76      4820      15.0  MUL_MAT q8_0 512x2048x1 cols=3x1
[ggml-cuda profile]      71.23   0.75       964      73.9  MUL_MAT q6_K 2048x8192x1 cols=3x1
[ggml-cuda profile]      64.98   0.69      4820      13.5  MUL_MAT q5_0 512x2048x1 cols=3x1
[ggml-cuda profile]      54.89   0.58      7470       7.3  SCALE
[ggml-cuda profile]      51.25   0.54      3177      16.1  CONT
[ggml-cuda profile]      50.04   0.53        30    1668.0  MUL_MAT q4_0 2048x4096x1 cols=2044x1
[ggml-cuda profile]      48.83   0.51        28    1743.9  MUL_MAT q4_0 4096x2048x1 cols=2044x1
[ggml-cuda profile]      47.80   0.50      7350       6.5  CPY
[ggml-cuda profile]      44.58   0.47      7350       6.1  ARANGE
[ggml-cuda profile]      43.31   0.46        82     528.1  MUL_MAT_ID q4_0 2048x512x256 cols=1x123
[ggml-cuda profile]      36.13   0.38      3177      11.4  ROPE
[ggml-cuda profile]      34.98   0.37       483      72.4  TOP_K
[ggml-cuda profile]      31.92   0.34      3177      10.0  ROPE +fused
[ggml-cuda profile]      25.59   0.27       483      53.0  MUL_MAT_ID q4_0 2048x512x256 cols=1x1 +fused
[ggml-cuda profile]      25.01   0.26        14    1786.4  MUL_MAT q8_0 4096x2048x1 cols=2044x1
[ggml-cuda profile]      20.76   0.22        60     346.1  MUL_MAT q8_0 2048x512x1 cols=2044x1
[ggml-cuda profile]      20.14   0.21        36     559.3  MUL_MAT_ID q4_0 512x2048x256 cols=8x123
[ggml-cuda profile]      20.02   0.21       482      41.5  MUL_MAT q4_0 2048x8192x1 cols=1x1
[ggml-cuda profile]      19.16   0.20         4    4790.2  MUL_MAT q6_K 2048x8192x1 cols=2044x1
[ggml-cuda profile]      16.90   0.18       482      35.1  MUL_MAT q4_0 4096x2048x1 cols=1x1
[ggml-cuda profile]      16.87   0.18       482      35.0  MUL_MAT q4_0 4096x2048x1 cols=1x1 +fused
[ggml-cuda profile]      16.52   0.17        60     275.4  MUL_MAT f32 2048x32x1 cols=123x1
[ggml-cuda profile]      16.45   0.17         5    3290.2  MUL_MAT_ID q4_1 512x2048x256 cols=8x2044
[ggml-cuda profile]      14.53   0.15        40     363.3  MUL_MAT q5_0 2048x512x1 cols=2044x1
[ggml-cuda profile]      14.45   0.15       483      29.9  MUL_MAT_ID q4_0 512x2048x256 cols=8x1
[ggml-cuda profile]      14.22   0.15        37     384.2  MUL_MAT q4_0 2048x8192x1 cols=2x1
```

_Terminé : 09:53 UTC, 137 s._
# Séance 04 — répétitions (bruit)

| variante | prompt | tokens prompt | tokens générés | tg t/s | brouillons acceptés/proposés |
|---|---|---|---|---|---|
| référence p02, MTP2 (3) | court | 2171 | 600 | **87.96** | 358/482 |
| référence p02, MTP2 (3) | long | 18861 | 600 | **71.37** | 331/535 |
| 04 liste 2, MTP2 (2) | court | 2171 | 600 | **91.27** | 351/494 |
| 04 liste 2, MTP2 (2) | long | 18861 | 600 | **77.64** | 329/539 |
| référence p02, MTP2 (4) | court | 2171 | 600 | **87.83** | 358/482 |
| référence p02, MTP2 (4) | long | 18861 | 600 | **71.46** | 331/535 |
| 04 liste 2, MTP2 (3) | court | 2171 | 600 | **94.09** | 351/494 |
| 04 liste 2, MTP2 (3) | long | 18861 | 600 | **76.74** | 329/539 |
| 04 liste 2, MTP n-max 1 | court | 2171 | 600 | **85.95** | 272/326 |
| 04 liste 2, MTP n-max 1 | long | 18861 | 600 | **73.6** | 256/342 |

_Terminé : 09:58 UTC._
