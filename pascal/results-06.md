# Fork Pascal — séance 06 (état du delta-net lu dans le cache, fin du bloc MoE fusionnée, micro-banc « bulles »)

- Date : 2026-11-11 11:23 UTC
- GPU : Tesla P40, 580.126.09, 24576 MiB
- Base : aa75ff3239f9cc675b081697deda9e1b112345c6 + 01 + 02 + 04 + 05 + 06 ; liste de tokens : /root/models/tokens2.txt
- Référence = même build, nouveautés coupées : `GGML_CUDA_PASCAL_GDN_DIRECT_STATE=0 GGML_CUDA_PASCAL_FUSE_SHEXP_GATE=0`

## 0. Micro-banc « bulles » : noyaux en série (A) contre noyau persistant (B)

### blocs par SM et lignes par prise : 16 8

- GPU : Tesla P40, sm_61, 30 SM, lancement coopératif : oui
- mode B : 480 blocs de 64 threads (16 par SM), 8 lignes par prise ; médiane de 15 passages

| scénario | étapes | Mo lus | A : µs/étape | A : Go/s | B : µs/étape | B : Go/s | A/B | résultats |
|---|---|---|---|---|---|---|---|---|
| qkv Q4_0 8192x2048 (9,4 Mo) x40 | 40 | 377.5 | 215.2 | 44 | 305.8 | 31 | 0.70 | identiques |
| z Q4_0 4096x2048 (4,7 Mo) x40 | 40 | 188.7 | 106.8 | 44 | 180.0 | 26 | 0.59 | identiques |
| descente experts 16384x512 (5,2 Mo) x40 | 40 | 209.7 | 211.4 | 25 | 307.6 | 17 | 0.69 | identiques |
| routeur F32 256x2048 (2,1 Mo) x40 | 40 | 83.9 | 70.4 | 30 | 541.3 | 4 | 0.13 | identiques |
| petit 512x2048 Q4_0 (0,6 Mo) x40 | 40 | 23.6 | 64.7 | 9 | 170.0 | 3 | 0.38 | identiques |
| portes 64x2048 F32 (0,5 Mo) x40 | 40 | 21.0 | 66.6 | 8 | 402.4 | 1 | 0.17 | identiques |
| minuscule 2 lignes (8 Ko) x200 | 200 | 1.6 | 17.0 | 0 | 42.7 | 0 | 0.40 | identiques |
| tête Q6_K 248320x2048 (417 Mo) x2 | 2 | 834.4 | 6617.1 | 63 | 7869.5 | 53 | 0.84 | identiques |
| qkv Q4_0 8192x2048 x40, 3 colonnes | 40 | 377.5 | 220.2 | 43 | 427.7 | 22 | 0.51 | identiques |
| token synthétique (40 couches + tête) | 962 | 2254.2 | 58.7 | 40 | 137.8 | 17 | 0.43 | identiques |
| token synthétique sans la tête | 961 | 1837.0 | 50.5 | 38 | 136.8 | 14 | 0.37 | identiques |
| vérification MTP synthétique, 3 colonnes | 1122 | 3428.6 | 106.5 | 29 | 279.3 | 11 | 0.38 | identiques |

Lecture : A/B > 1 = le noyau persistant est plus rapide. « Go/s » = octets de poids lus / temps total.

### blocs par SM et lignes par prise : 32 8

> bulles 32 8 : échec (voir /root/fork6/logs/bulles.err : )

### blocs par SM et lignes par prise : 16 2

- GPU : Tesla P40, sm_61, 30 SM, lancement coopératif : oui
- mode B : 480 blocs de 64 threads (16 par SM), 2 lignes par prise ; médiane de 15 passages

| scénario | étapes | Mo lus | A : µs/étape | A : Go/s | B : µs/étape | B : Go/s | A/B | résultats |
|---|---|---|---|---|---|---|---|---|
| qkv Q4_0 8192x2048 (9,4 Mo) x40 | 40 | 377.5 | 99.2 | 95 | 106.1 | 89 | 0.93 | identiques |
| z Q4_0 4096x2048 (4,7 Mo) x40 | 40 | 188.7 | 51.1 | 92 | 58.9 | 80 | 0.87 | identiques |
| descente experts 16384x512 (5,2 Mo) x40 | 40 | 209.7 | 95.2 | 55 | 102.7 | 51 | 0.93 | identiques |
| routeur F32 256x2048 (2,1 Mo) x40 | 40 | 83.9 | 14.1 | 149 | 72.0 | 29 | 0.20 | identiques |
| petit 512x2048 Q4_0 (0,6 Mo) x40 | 40 | 23.6 | 8.5 | 70 | 19.1 | 31 | 0.44 | identiques |
| portes 64x2048 F32 (0,5 Mo) x40 | 40 | 21.0 | 10.2 | 51 | 73.8 | 7 | 0.14 | identiques |
| minuscule 2 lignes (8 Ko) x200 | 200 | 1.6 | 4.1 | 2 | 18.3 | 0 | 0.23 | identiques |
| tête Q6_K 248320x2048 (417 Mo) x2 | 2 | 834.4 | 3098.3 | 135 | 3107.4 | 134 | 1.00 | identiques |
| qkv Q4_0 8192x2048 x40, 3 colonnes | 40 | 377.5 | 104.4 | 90 | 116.2 | 81 | 0.90 | identiques |
| token synthétique (40 couches + tête) | 962 | 2254.2 | 26.7 | 88 | 42.1 | 56 | 0.63 | identiques |
| token synthétique sans la tête | 961 | 1837.0 | 23.5 | 82 | 39.4 | 49 | 0.60 | identiques |
| vérification MTP synthétique, 3 colonnes | 1122 | 3428.6 | 49.4 | 62 | 70.4 | 43 | 0.70 | identiques |

Lecture : A/B > 1 = le noyau persistant est plus rapide. « Go/s » = octets de poids lus / temps total.

### blocs par SM et lignes par prise : 8 8

- GPU : Tesla P40, sm_61, 30 SM, lancement coopératif : oui
- mode B : 240 blocs de 64 threads (8 par SM), 8 lignes par prise ; médiane de 15 passages

| scénario | étapes | Mo lus | A : µs/étape | A : Go/s | B : µs/étape | B : Go/s | A/B | résultats |
|---|---|---|---|---|---|---|---|---|
| qkv Q4_0 8192x2048 (9,4 Mo) x40 | 40 | 377.5 | 99.2 | 95 | 108.0 | 87 | 0.92 | identiques |
| z Q4_0 4096x2048 (4,7 Mo) x40 | 40 | 188.7 | 51.1 | 92 | 62.9 | 75 | 0.81 | identiques |
| descente experts 16384x512 (5,2 Mo) x40 | 40 | 209.7 | 95.2 | 55 | 107.6 | 49 | 0.88 | identiques |
| routeur F32 256x2048 (2,1 Mo) x40 | 40 | 83.9 | 14.1 | 149 | 68.1 | 31 | 0.21 | identiques |
| petit 512x2048 Q4_0 (0,6 Mo) x40 | 40 | 23.6 | 8.5 | 70 | 20.5 | 29 | 0.41 | identiques |
| portes 64x2048 F32 (0,5 Mo) x40 | 40 | 21.0 | 10.2 | 51 | 68.7 | 8 | 0.15 | identiques |
| minuscule 2 lignes (8 Ko) x200 | 200 | 1.6 | 4.1 | 2 | 9.8 | 1 | 0.42 | identiques |
| tête Q6_K 248320x2048 (417 Mo) x2 | 2 | 834.4 | 3098.2 | 135 | 3103.3 | 134 | 1.00 | identiques |
| qkv Q4_0 8192x2048 x40, 3 colonnes | 40 | 377.5 | 104.4 | 90 | 136.0 | 69 | 0.77 | identiques |
| token synthétique (40 couches + tête) | 962 | 2254.2 | 26.7 | 88 | 35.9 | 65 | 0.74 | identiques |
| token synthétique sans la tête | 961 | 1837.0 | 23.5 | 81 | 33.3 | 57 | 0.71 | identiques |
| vérification MTP synthétique, 3 colonnes | 1122 | 3428.6 | 49.4 | 62 | 78.1 | 39 | 0.63 | identiques |

Lecture : A/B > 1 = le noyau persistant est plus rapide. « Go/s » = octets de poids lus / temps total.

## 1. Compilation

- compilé en 438 s

## 2. Justesse

- test-backend-ops GATED_DELTA_NET, GET_ROWS : 259/259 tests passed (code 0)
- sortie déterministe (200 tokens, temp 0), variante etat : identique à la référence (822 octets)
- sortie déterministe (200 tokens, temp 0), variante porte : diffère de la référence à l'octet 313 sur 822
- sortie déterministe (200 tokens, temp 0), variante tout : diffère de la référence à l'octet 313 sur 822
  (attendu : « etat » identique au bit près, la lecture directe ne change aucun calcul ; « porte » peut diverger, l'ordre des sommes change)
- perplexité, 16 blocs de 512 tokens (master : 7.1616 ± 0.28264) : référence 7.1616 ± 0.28264 ; 06 7.1616 ± 0.28264
- perplexité, 120 blocs de 64 tokens (séance 05 : 12.7132 / 12.7137 ; exerce la fusion de fin de bloc MoE) : référence 12.7113 ± 0.60031 ; 06 12.7113 ± 0.60031

## 3. Profil par type de nœud (llama-bench tg, 1 token à la fois, sans CUDA graphs)

Attendu avec 06 : GET_ROWS passe de 62 à 32 appels par token (les 30 copies d'état de 2 Mo disparaissent),
et une ligne `MUL_MAT f32 2048x1x1 cols=1x1 +fused` apparaît (40 appels par token).

### référence

```
[ggml-cuda profile] 97 graphs, 1896.9 ms in nodes
[ggml-cuda profile]         ms      %     count   us/call  node
[ggml-cuda profile]     190.00  10.02      3880      49.0  MUL_MAT_ID q4_0 2048x512x256 cols=1x1 +fused
[ggml-cuda profile]     163.74   8.63      3492      46.9  MUL_MAT q4_0 2048x8192x1 cols=1x1
[ggml-cuda profile]     139.81   7.37        97    1441.4  MUL_MAT q6_K 2048x248320x1 cols=1x1
[ggml-cuda profile]     134.41   7.09     15617       8.6  RMS_NORM +fused
[ggml-cuda profile]      95.01   5.01      2910      32.6  MUL_MAT q4_0 2048x4096x1 cols=1x1
[ggml-cuda profile]      92.96   4.90      3395      27.4  MUL_MAT_ID q4_0 512x2048x256 cols=8x1
[ggml-cuda profile]      90.85   4.79      6014      15.1  GET_ROWS
[ggml-cuda profile]      77.34   4.08      2425      31.9  MUL_MAT q4_0 4096x2048x1 cols=1x1 +fused
[ggml-cuda profile]      74.87   3.95      2910      25.7  GATED_DELTA_NET +fused
[ggml-cuda profile]      74.45   3.92      3880      19.2  MUL_MAT f32 2048x256x1 cols=1x1
[ggml-cuda profile]      64.36   3.39      3880      16.6  SOFT_MAX +fused
[ggml-cuda profile]      62.61   3.30      1358      46.1  MUL_MAT q8_0 4096x2048x1 cols=1x1 +fused
[ggml-cuda profile]      44.02   2.32      1940      22.7  MUL_MAT q8_0 2048x512x1 cols=1x1 +fused
[ggml-cuda profile]      43.51   2.29      2910      15.0  MUL_MAT f32 2048x32x1 cols=1x1 +fused
[ggml-cuda profile]      43.31   2.28      1940      22.3  MUL_MAT q5_0 2048x512x1 cols=1x1 +fused
[ggml-cuda profile]      43.18   2.28      3880      11.1  MUL +fused
[ggml-cuda profile]      38.49   2.03      3880       9.9  MUL
[ggml-cuda profile]      38.40   2.02      2910      13.2  CPY
[ggml-cuda profile]      36.16   1.91      3880       9.3  MUL_MAT f32 2048x1x1 cols=1x1
[ggml-cuda profile]      33.12   1.75      3880       8.5  UNARY +fused
[ggml-cuda profile]      33.09   1.74      1940      17.1  MUL_MAT q8_0 512x2048x1 cols=1x1
[ggml-cuda profile]      32.94   1.74      3880       8.5  ADD +fused
GET_ROWS : 6014 appels, 15.1 us/appel
nœuds par token (somme des appels / 97 graphes) : 1036
```

### 06

```
[ggml-cuda profile] 97 graphs, 1769.6 ms in nodes
[ggml-cuda profile]         ms      %     count   us/call  node
[ggml-cuda profile]     190.53  10.77      3880      49.1  MUL_MAT_ID q4_0 2048x512x256 cols=1x1 +fused
[ggml-cuda profile]     164.03   9.27      3492      47.0  MUL_MAT q4_0 2048x8192x1 cols=1x1
[ggml-cuda profile]     139.90   7.91        97    1442.3  MUL_MAT q6_K 2048x248320x1 cols=1x1
[ggml-cuda profile]     137.29   7.76     15617       8.8  RMS_NORM +fused
[ggml-cuda profile]      94.63   5.35      3395      27.9  MUL_MAT_ID q4_0 512x2048x256 cols=8x1
[ggml-cuda profile]      94.09   5.32      2910      32.3  MUL_MAT q4_0 2048x4096x1 cols=1x1
[ggml-cuda profile]      78.61   4.44      2425      32.4  MUL_MAT q4_0 4096x2048x1 cols=1x1 +fused
[ggml-cuda profile]      78.39   4.43      2910      26.9  GATED_DELTA_NET +fused
[ggml-cuda profile]      76.00   4.29      3880      19.6  MUL_MAT f32 2048x256x1 cols=1x1
[ggml-cuda profile]      64.67   3.65      3880      16.7  SOFT_MAX +fused
[ggml-cuda profile]      63.40   3.58      1358      46.7  MUL_MAT q8_0 4096x2048x1 cols=1x1 +fused
[ggml-cuda profile]      44.47   2.51      1940      22.9  MUL_MAT q8_0 2048x512x1 cols=1x1 +fused
[ggml-cuda profile]      44.47   2.51      3880      11.5  MUL_MAT f32 2048x1x1 cols=1x1 +fused
[ggml-cuda profile]      43.71   2.47      1940      22.5  MUL_MAT q5_0 2048x512x1 cols=1x1 +fused
[ggml-cuda profile]      43.40   2.45      3880      11.2  MUL +fused
[ggml-cuda profile]      41.69   2.36      2910      14.3  MUL_MAT f32 2048x32x1 cols=1x1 +fused
[ggml-cuda profile]      38.54   2.18      2910      13.2  CPY
[ggml-cuda profile]      33.85   1.91      3880       8.7  UNARY +fused
[ggml-cuda profile]      33.38   1.89      1940      17.2  MUL_MAT q8_0 512x2048x1 cols=1x1
[ggml-cuda profile]      32.70   1.85       970      33.7  FLASH_ATTN_EXT
[ggml-cuda profile]      32.41   1.83      3104      10.4  GET_ROWS
[ggml-cuda profile]      30.43   1.72      1940      15.7  MUL_MAT q5_0 512x2048x1 cols=1x1
GET_ROWS : 3104 appels, 10.4 us/appel
nœuds par token (somme des appels / 97 graphes) : 886
```

## 4. llama-bench tg128 (-fa on, 3 répétitions)

| variante | tg128 t/s |
|---|---|
| référence | 71.61 ± 0.08 |
| état direct seul | 73.87 ± 0.08 |
| fin de bloc MoE seule | 73.26 ± 0.09 |
| 06 (les deux) | 75.74 ± 0.09 |
| référence (2e passage) | 71.61 ± 0.13 |
| 06 (2e passage) | 75.82 ± 0.05 |

## 5. Requêtes réelles de vérification (llama-server, /v1/chat/completions, temp 0, liste de tokens)

t/s = somme des tokens générés / somme des temps de génération ; « textes identiques » : par rapport à la première ligne.

| variante | réponses | tokens générés | tg t/s (somme) | brouillons acceptés/proposés | textes identiques | prompt : t/s, temps total des 8 requêtes | t/s par requête |
|---|---|---|---|---|---|---|---|
| référence (05), MTP2 | 8/8 | 2141 | **100.9** | 1306/1656 | 8/8 | 1455 t/s, 36.8 s | 100 100 98 97 105 100 103 101 |
| 06, MTP2 | 8/8 | 2119 | **106.3** | 1305/1616 | 3/8 | 1454 t/s, 35.6 s | 103 103 103 106 109 106 110 104 |
| 06 état direct seul, MTP2 | 8/8 | 2141 | **103.6** | 1306/1656 | 8/8 | 1458 t/s, 36.3 s | 102 102 101 101 108 103 105 103 |
| référence (2e passage), MTP2 | 8/8 | 2141 | **101.5** | 1306/1656 | 8/8 | 1450 t/s, 36.8 s | 99 101 100 98 106 101 104 99 |
| 06 (2e passage), MTP2 | 8/8 | 2119 | **106.6** | 1305/1616 | 3/8 | 1458 t/s, 35.5 s | 103 104 104 107 109 106 110 104 |
| référence, sans spéculatif | 8/8 | 2141 | **67.6** | 0/0 | 8/8 | 1530 t/s, 46.5 s | 67 67 67 67 68 67 67 67 |
| 06, sans spéculatif | 8/8 | 2119 | **71.4** | 0/0 | 3/8 | 1527 t/s, 44.6 s | 71 71 71 71 71 71 71 71 |

## 6. Listes de nœuds

- llama-bench, 1 token : 3787 nœuds, /root/web/graphe06-bench.txt
- serveur + MTP n-max 2, 40e graphe : 3908 nœuds, /root/web/graphe06-serveur-mtp.txt ; première ligne MUL_MAT :    13 MUL_MAT            f32    [8192,3,1,1]  'node_15' <- leaf(blk.0.attn_qkv.weight q4_0

_Terminé : 2026-11-11 11:45 UTC, durée 1345 s. Journaux dans /root/fork6/logs._
