# Fork Pascal — séance 05 (fusion des petits noyaux du delta-net)

- Date : 2026-11-11 10:20 UTC
- GPU : Tesla P40, 580.126.09, 24576 MiB
- Base : aa75ff3239f9cc675b081697deda9e1b112345c6 + 01 + 02 + 04 + 05 ; liste de tokens : /root/models/tokens2.txt
- Référence = même build, fusions coupées : `LLAMA_PASCAL_GDN_QK_NORM=0 GGML_CUDA_PASCAL_FUSE_GDN_GATES=0`

## 1. Compilation

- compilé en 449 s

## 2. Liste des nœuds d'un graphe de génération (1 token)

- ref : 3817 nœuds, fichier /root/web/graphe05-ref.txt
- 05 : 3787 nœuds, fichier /root/web/graphe05-05.txt

## 3. Justesse

- test-backend-ops RMS_NORM, SCALE, GATED_DELTA_NET : 115/115 tests passed (code 0)
- perplexité (master en séance 1 : 7.1616 ± 0.28264) : référence 7.1616 ± 0.28264 ; 05 7.1616 ± 0.28264
  (la perplexité évalue des lots de 512 tokens : elle couvre la norme q/k commune, pas la fusion des portes, limitée à 64 tokens)
- sortie déterministe (200 tokens, temp 0), variante qk : identique à la référence (708 octets)
- sortie déterministe (200 tokens, temp 0), variante portes : diffère de la référence à l'octet 39 sur 708
- sortie déterministe (200 tokens, temp 0), variante tout : diffère de la référence à l'octet 39 sur 708
  (la norme q/k commune doit être identique au bit près ; la fusion des portes change l'ordre des sommes, une divergence tardive est possible)

## 4. Profil par type de nœud (llama-bench tg, 1 token à la fois, sans CUDA graphs)

La fusion des portes est active si la ligne `MUL_MAT f32 2048x32x1 cols=1x1 +fused` apparaît (30 appels par token).

### référence

```
[ggml-cuda profile] 97 graphs, 2060.0 ms in nodes
[ggml-cuda profile]         ms      %     count   us/call  node
[ggml-cuda profile]     194.05   9.42      3880      50.0  MUL_MAT_ID q4_0 2048x512x256 cols=1x1 +fused
[ggml-cuda profile]     167.12   8.11      3492      47.9  MUL_MAT q4_0 2048x8192x1 cols=1x1
[ggml-cuda profile]     164.08   7.97     18527       8.9  RMS_NORM +fused
[ggml-cuda profile]     139.98   6.80        97    1443.1  MUL_MAT q6_K 2048x248320x1 cols=1x1
[ggml-cuda profile]      97.37   4.73      2910      33.5  MUL_MAT q4_0 2048x4096x1 cols=1x1
[ggml-cuda profile]      96.08   4.66      3395      28.3  MUL_MAT_ID q4_0 512x2048x256 cols=8x1
[ggml-cuda profile]      92.36   4.48      6014      15.4  GET_ROWS
[ggml-cuda profile]      80.20   3.89      2425      33.1  MUL_MAT q4_0 4096x2048x1 cols=1x1 +fused
[ggml-cuda profile]      75.70   3.67      2910      26.0  GATED_DELTA_NET +fused
[ggml-cuda profile]      71.42   3.47      5820      12.3  MUL_MAT f32 2048x32x1 cols=1x1
[ggml-cuda profile]      64.61   3.14      3880      16.7  SOFT_MAX +fused
[ggml-cuda profile]      64.02   3.11      1358      47.1  MUL_MAT q8_0 4096x2048x1 cols=1x1 +fused
[ggml-cuda profile]      62.98   3.06      3880      16.2  MUL_MAT f32 2048x256x1 cols=1x1
[ggml-cuda profile]      59.52   2.89      6790       8.8  UNARY +fused
[ggml-cuda profile]      45.70   2.22      3880      11.8  MUL +fused
[ggml-cuda profile]      45.63   2.21      1940      23.5  MUL_MAT q8_0 2048x512x1 cols=1x1 +fused
[ggml-cuda profile]      45.49   2.21      6790       6.7  UNARY
[ggml-cuda profile]      45.15   2.19      1940      23.3  MUL_MAT q5_0 2048x512x1 cols=1x1 +fused
[ggml-cuda profile]      39.45   1.92      2910      13.6  CPY
[ggml-cuda profile]      36.66   1.78      3880       9.4  MUL_MAT f32 2048x1x1 cols=1x1
[ggml-cuda profile]      35.42   1.72      3880       9.1  MUL
[ggml-cuda profile]      34.68   1.68      3880       8.9  ADD +fused
[ggml-cuda profile]      34.49   1.67      1940      17.8  MUL_MAT q8_0 512x2048x1 cols=1x1
[ggml-cuda profile]      33.55   1.63      3007      11.2  ADD
[ggml-cuda profile]      33.52   1.63       970      34.6  FLASH_ATTN_EXT
nœuds par token (somme des appels / 97 graphes) : 1186
```

### 05

```
[ggml-cuda profile] 97 graphs, 1891.0 ms in nodes
[ggml-cuda profile]         ms      %     count   us/call  node
[ggml-cuda profile]     189.92  10.04      3880      48.9  MUL_MAT_ID q4_0 2048x512x256 cols=1x1 +fused
[ggml-cuda profile]     165.73   8.76      3492      47.5  MUL_MAT q4_0 2048x8192x1 cols=1x1
[ggml-cuda profile]     139.80   7.39        97    1441.3  MUL_MAT q6_K 2048x248320x1 cols=1x1
[ggml-cuda profile]     134.44   7.11     15617       8.6  RMS_NORM +fused
[ggml-cuda profile]      94.81   5.01      2910      32.6  MUL_MAT q4_0 2048x4096x1 cols=1x1
[ggml-cuda profile]      92.58   4.90      3395      27.3  MUL_MAT_ID q4_0 512x2048x256 cols=8x1
[ggml-cuda profile]      88.66   4.69      6014      14.7  GET_ROWS
[ggml-cuda profile]      78.19   4.13      2425      32.2  MUL_MAT q4_0 4096x2048x1 cols=1x1 +fused
[ggml-cuda profile]      74.94   3.96      3880      19.3  MUL_MAT f32 2048x256x1 cols=1x1
[ggml-cuda profile]      72.77   3.85      2910      25.0  GATED_DELTA_NET +fused
[ggml-cuda profile]      62.72   3.32      1358      46.2  MUL_MAT q8_0 4096x2048x1 cols=1x1 +fused
[ggml-cuda profile]      61.91   3.27      3880      16.0  SOFT_MAX +fused
[ggml-cuda profile]      43.68   2.31      1940      22.5  MUL_MAT q8_0 2048x512x1 cols=1x1 +fused
[ggml-cuda profile]      42.88   2.27      1940      22.1  MUL_MAT q5_0 2048x512x1 cols=1x1 +fused
[ggml-cuda profile]      42.53   2.25      3880      11.0  MUL +fused
[ggml-cuda profile]      42.32   2.24      2910      14.5  MUL_MAT f32 2048x32x1 cols=1x1 +fused
[ggml-cuda profile]      38.80   2.05      3880      10.0  MUL
[ggml-cuda profile]      38.34   2.03      2910      13.2  CPY
[ggml-cuda profile]      36.72   1.94      3880       9.5  MUL_MAT f32 2048x1x1 cols=1x1
[ggml-cuda profile]      33.50   1.77      1940      17.3  MUL_MAT q8_0 512x2048x1 cols=1x1
[ggml-cuda profile]      32.87   1.74       970      33.9  FLASH_ATTN_EXT
[ggml-cuda profile]      32.67   1.73      3880       8.4  ADD +fused
[ggml-cuda profile]      32.59   1.72      3880       8.4  UNARY +fused
[ggml-cuda profile]      30.14   1.59      1940      15.5  MUL_MAT q5_0 512x2048x1 cols=1x1
[ggml-cuda profile]      28.97   1.53      2910      10.0  CONCAT
nœuds par token (somme des appels / 97 graphes) : 1036
```

## 5. llama-bench tg128 (-fa on, 3 répétitions)

| variante | tg128 t/s |
|---|---|
| référence | 70.21 ± 0.05 |
| norme q/k seule | 70.40 ± 0.05 |
| portes seules | 71.41 ± 0.05 |
| 05 (les deux) | 71.69 ± 0.05 |
| référence (2e passage) | 70.13 ± 0.05 |
| 05 (2e passage) | 71.62 ± 0.03 |

## 6. Génération réelle (llama-server, q40, -fa on, -c 40000, 600 tokens, temp 0)

| variante | prompt | tokens prompt | tokens générés | tg t/s | brouillons acceptés/proposés |
|---|---|---|---|---|---|
| référence (04 + liste), MTP2 | court | 2171 | 600 | **94.45** | 351/494 |
| référence (04 + liste), MTP2 | long | 18861 | 600 | **77.37** | 329/539 |
| 05 + liste, MTP2 | court | 2171 | 600 | **87.81** | 356/484 |
| 05 + liste, MTP2 | long | 18861 | 600 | **83.37** | 343/512 |
| référence (2e passage), MTP2 | court | 2171 | 600 | **94.73** | 351/494 |
| référence (2e passage), MTP2 | long | 18861 | 600 | **74.3** | 329/539 |
| 05 + liste (2e passage), MTP2 | court | 2171 | 600 | **97.8** | 356/484 |
| 05 + liste (2e passage), MTP2 | long | 18861 | 600 | **82.87** | 343/512 |
| 05 norme q/k seule + liste, MTP2 | court | 2171 | 600 | **87.57** | 351/494 |
| 05 norme q/k seule + liste, MTP2 | long | 18861 | 600 | **77.8** | 329/539 |
| référence, sans MTP | court | 2171 | 600 | **66.26** | —/— |
| référence, sans MTP | long | 18861 | 600 | **60.04** | —/— |
| 05, sans MTP | court | 2171 | 600 | **67.76** | —/— |
| 05, sans MTP | long | 18861 | 600 | **61.26** | —/— |

_Terminé : 2026-11-11 10:36 UTC, durée 982 s. Journaux dans /root/fork5/logs._
# Perplexité par blocs de 64 tokens (120 blocs) : exerce la fusion des portes
- référence (fusions coupées) : 0.20.184.896 I Final estimate: PPL = 12.7132 +/- 0.60053
- portes seules : 0.19.608.906 I Final estimate: PPL = 12.7137 +/- 0.60065
- 05 (les deux) : 0.18.572.403 I Final estimate: PPL = 12.7137 +/- 0.60065
## sorties déterministes
### ref
 Jean-Pierre Lefèvre, 50 ans, est un homme d'expérience, calme et méthodique. Il est marié et père de deux enfants. Il est marié à Marie, une enseignante de lettres classiques, et a deux enfants, Luc (17 ans) et Emma (14 ans). Il est marié à Marie depuis 20 ans. Il est marié à Marie depuis 20 ans. Il est marié à Marie depuis 20 ans. Il est marié à Marie depuis 20 ans. Il est marié à Marie depuis 20 a
### portes
 Jean-Pierre Lefèvre, 50 ans, est un flic de la brigade criminelle de Lyon. Il est marié et a deux enfants. Il est très attaché à sa ville et à ses habitants. Il est très attaché à sa ville et à ses habitants. Il est très attaché à sa ville et à ses habitants. Il est très attaché à sa ville et ses habitants. Il est très attaché à sa ville et à ses habitants. Il est très attaché à sa ville et �
### tout
 Jean-Pierre Lefèvre, 50 ans, est un flic de la brigade criminelle de Lyon. Il est marié et a deux enfants. Il est très attaché à sa ville et à ses habitants. Il est très attaché à sa ville et à ses habitants. Il est très attaché à sa ville et à ses habitants. Il est très attaché à sa ville et ses habitants. Il est très attaché à sa ville et à ses habitants. Il est très attaché à sa ville et �
