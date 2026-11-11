# Fork Pascal (P40, sm_61) — résultats

- Date : 2026-11-11 00:39 UTC
- Commit de base : aa75ff3239f9cc675b081697deda9e1b112345c6
- GPU : Tesla P40, 580.126.09, 24576 MiB
- nvcc : Build cuda_12.4.r12.4/compiler.34097967_0
- Noyau : 5.15.0-171-generic — Ubuntu 22.04.4 LTS
- Builds : master = amont ; p01 = + PR #27721 #25479 #31352 ; p02 = p01 + optimisations du fork ; p03 = p02 + 03-mtp.patch (coût d'un pas MTP)

## 1. Compilation

- master : compilé en 771 s
- master : compilé en 713 s
- p01 : **ÉCHEC de compilation** (fin de /root/fork/logs/build-p01.log) :
```
gmake: Makefile: No such file or directory
gmake: *** No rule to make target 'Makefile'.  Stop.
```
- p01 : **ÉCHEC de compilation** (fin de /root/fork/logs/build-p01.log) :
```
gmake: Makefile: No such file or directory
gmake: *** No rule to make target 'Makefile'.  Stop.
```
- p02 : **ÉCHEC de compilation** (fin de /root/fork/logs/build-p02.log) :
```
gmake: Makefile: No such file or directory
gmake: *** No rule to make target 'Makefile'.  Stop.
```

- p02 : compilé en 472 s
- p03 : compilé en 475 s

## 2. Justesse

### 2.1 test-backend-ops (MUL_MAT, MUL_MAT_ID contre le CPU)

| build | variante | résultat | code |
|---|---|---|---|
| master | defaut | 3484/3484 tests passed | OK |
| p02 | defaut | 3484/3484 tests passed | OK |
| p02 | amont | 3484/3484 tests passed | OK |
| p02 | rows0 | 3484/3484 tests passed | OK |
| p02 | rows1 | 3484/3484 tests passed | OK |
| p02 | rows2 | 3484/3484 tests passed | OK |
| p02 | rows4 | 3484/3484 tests passed | OK |
| p02 | multi0 | 3484/3484 tests passed | OK |
| p02 | q8cache0 | 3484/3484 tests passed | OK |
| p02 | sans-graphs | 3484/3484 tests passed | OK |
| p02 | graph-opt | 3484/3484 tests passed | OK |
| p02 | graph-1bloc | 3484/3484 tests passed | OK |
| p02 | rows4+1bloc | 3484/3484 tests passed | OK |

### 2.2 Sortie déterministe (temp 0, seed 42, 200 tokens) comparée à master, et perplexité courte

Une divergence n'est pas forcément un bug : l'ordre des sommes flottantes change avec les noyaux.
Dans ce cas, la perplexité doit rester égale à celle de master à l'incertitude près.

| quant | build | variante | sortie vs master | PPL |
|---|---|---|---|---|
| q40 | master | defaut | (référence, 769 octets) | 7.1616 ± 0.28264 |
| q40 | p02 | defaut | diffère à l'octet ? sur 769 (≈ token 0) | 7.1616 ± 0.28264 |
| q40 | p02 | amont | diffère à l'octet ? sur 769 (≈ token 0) | 7.1616 ± 0.28264 |
| q40 | p02 | rows0 | diffère à l'octet ? sur 769 (≈ token 0) | — |
| q40 | p02 | rows1 | diffère à l'octet ? sur 769 (≈ token 0) | 7.1616 ± 0.28264 |
| q40 | p02 | rows2 | diffère à l'octet ? sur 769 (≈ token 0) | — |
| q40 | p02 | rows4 | diffère à l'octet ? sur 769 (≈ token 0) | 7.1616 ± 0.28264 |
| q40 | p02 | multi0 | diffère à l'octet ? sur 769 (≈ token 0) | — |
| q40 | p02 | q8cache0 | diffère à l'octet ? sur 769 (≈ token 0) | 7.1616 ± 0.28264 |
| q40 | p02 | sans-graphs | diffère à l'octet ? sur 769 (≈ token 0) | — |
| q40 | p02 | graph-opt | diffère à l'octet ? sur 769 (≈ token 0) | — |
| q40 | p02 | graph-1bloc | diffère à l'octet ? sur 769 (≈ token 0) | — |
| q40 | p02 | rows4+1bloc | diffère à l'octet ? sur 769 (≈ token 0) | — |
| q4ks | master | defaut | (référence, 778 octets) | 7.1147 ± 0.27911 |
| q4ks | p02 | defaut | diffère à l'octet ? sur 778 (≈ token 0) | 7.1147 ± 0.27911 |
| q4ks | p02 | amont | diffère à l'octet ? sur 778 (≈ token 0) | 7.1147 ± 0.27911 |
| q4ks | p02 | rows0 | diffère à l'octet ? sur 778 (≈ token 0) | — |
| q4ks | p02 | rows1 | diffère à l'octet ? sur 778 (≈ token 0) | 7.1147 ± 0.27911 |
| q4ks | p02 | rows2 | diffère à l'octet ? sur 778 (≈ token 0) | — |
| q4ks | p02 | rows4 | diffère à l'octet ? sur 778 (≈ token 0) | 7.1147 ± 0.27911 |
| q4ks | p02 | multi0 | diffère à l'octet ? sur 778 (≈ token 0) | — |
| q4ks | p02 | q8cache0 | diffère à l'octet ? sur 778 (≈ token 0) | 7.1147 ± 0.27911 |
| q4ks | p02 | sans-graphs | diffère à l'octet ? sur 778 (≈ token 0) | — |
| q4ks | p02 | graph-opt | diffère à l'octet ? sur 778 (≈ token 0) | — |
| q4ks | p02 | graph-1bloc | diffère à l'octet ? sur 778 (≈ token 0) | — |
| q4ks | p02 | rows4+1bloc | diffère à l'octet ? sur 778 (≈ token 0) | — |

## 3. llama-bench (pp4096 / tg128, -ub 2048, 3 répétitions)

| build | variante | quant | options | test | t/s |
|---|---|---|---|---|---|
| master | defaut | q40 | -fa off -ub 2048 -p 4096 -n 128 -d 0 | pp4096 | 1578.23 ± 3.12 |
| master | defaut | q40 | -fa off -ub 2048 -p 4096 -n 128 -d 0 | tg128 | 63.71 ± 0.03 |
| master | defaut | q40 | -fa off -ub 2048 -p 4096 -n 128 -d 32768 | échec | llama_bench: error: failed to create context with model '/root/models/q40.gguf' |
| master | defaut | q40 | -fa on -ub 2048 -p 4096 -n 128 -d 32768 | pp4096 @ d32768 | 534.89 ± 0.04 |
| master | defaut | q40 | -fa on -ub 2048 -p 4096 -n 128 -d 32768 | tg128 @ d32768 | 53.84 ± 0.05 |
| master | defaut | q40 | -fa on -ctk q8_0 -ctv q8_0 -ub 2048 -p 4096 -n 128 -d 32768 | pp4096 @ d32768 | 533.78 ± 0.30 |
| master | defaut | q40 | -fa on -ctk q8_0 -ctv q8_0 -ub 2048 -p 4096 -n 128 -d 32768 | tg128 @ d32768 | 44.67 ± 0.05 |
| p02 | defaut | q40 | -fa off -ub 2048 -p 4096 -n 128 -d 0 | pp4096 | 1791.55 ± 6.10 |
| p02 | defaut | q40 | -fa off -ub 2048 -p 4096 -n 128 -d 0 | tg128 | 68.19 ± 0.40 |
| p02 | defaut | q40 | -fa off -ub 2048 -p 4096 -n 128 -d 32768 | échec | llama_bench: error: failed to create context with model '/root/models/q40.gguf' |
| p02 | defaut | q40 | -fa on -ub 2048 -p 4096 -n 128 -d 32768 | pp4096 @ d32768 | 556.91 ± 1.10 |
| p02 | defaut | q40 | -fa on -ub 2048 -p 4096 -n 128 -d 32768 | tg128 @ d32768 | 57.28 ± 0.09 |
| p02 | defaut | q40 | -fa on -ctk q8_0 -ctv q8_0 -ub 2048 -p 4096 -n 128 -d 32768 | pp4096 @ d32768 | 556.17 ± 0.16 |
| p02 | defaut | q40 | -fa on -ctk q8_0 -ctv q8_0 -ub 2048 -p 4096 -n 128 -d 32768 | tg128 @ d32768 | 47.57 ± 0.05 |
| p03 | defaut | q40 | -fa off -ub 2048 -p 4096 -n 128 -d 0 | pp4096 | 1797.11 ± 2.33 |
| p03 | defaut | q40 | -fa off -ub 2048 -p 4096 -n 128 -d 0 | tg128 | 68.67 ± 0.02 |
| p03 | defaut | q40 | -fa off -ub 2048 -p 4096 -n 128 -d 32768 | échec | llama_bench: error: failed to create context with model '/root/models/q40.gguf' |
| p03 | defaut | q40 | -fa on -ub 2048 -p 4096 -n 128 -d 32768 | pp4096 @ d32768 | 556.76 ± 0.55 |
| p03 | defaut | q40 | -fa on -ub 2048 -p 4096 -n 128 -d 32768 | tg128 @ d32768 | 57.19 ± 0.20 |
| p03 | defaut | q40 | -fa on -ctk q8_0 -ctv q8_0 -ub 2048 -p 4096 -n 128 -d 32768 | pp4096 @ d32768 | 556.33 ± 0.60 |
| p03 | defaut | q40 | -fa on -ctk q8_0 -ctv q8_0 -ub 2048 -p 4096 -n 128 -d 32768 | tg128 @ d32768 | 47.66 ± 0.05 |
| master | defaut | q4ks | -fa off -ub 2048 -p 4096 -n 128 -d 0 | pp4096 | 1539.82 ± 0.61 |
| master | defaut | q4ks | -fa off -ub 2048 -p 4096 -n 128 -d 0 | tg128 | 56.14 ± 0.04 |
| master | defaut | q4ks | -fa off -ub 2048 -p 4096 -n 128 -d 32768 | échec | llama_bench: error: failed to create context with model '/root/models/q4ks.gguf' |
| master | defaut | q4ks | -fa on -ub 2048 -p 4096 -n 128 -d 32768 | pp4096 @ d32768 | 528.42 ± 0.22 |
| master | defaut | q4ks | -fa on -ub 2048 -p 4096 -n 128 -d 32768 | tg128 @ d32768 | 48.29 ± 0.04 |
| master | defaut | q4ks | -fa on -ctk q8_0 -ctv q8_0 -ub 2048 -p 4096 -n 128 -d 32768 | pp4096 @ d32768 | 527.79 ± 0.25 |
| master | defaut | q4ks | -fa on -ctk q8_0 -ctv q8_0 -ub 2048 -p 4096 -n 128 -d 32768 | tg128 @ d32768 | 41.26 ± 0.02 |
| p02 | defaut | q4ks | -fa off -ub 2048 -p 4096 -n 128 -d 0 | pp4096 | 1629.69 ± 2.81 |
| p02 | defaut | q4ks | -fa off -ub 2048 -p 4096 -n 128 -d 0 | tg128 | 60.21 ± 0.03 |
| p02 | defaut | q4ks | -fa off -ub 2048 -p 4096 -n 128 -d 32768 | échec | llama_bench: error: failed to create context with model '/root/models/q4ks.gguf' |
| p02 | defaut | q4ks | -fa on -ub 2048 -p 4096 -n 128 -d 32768 | pp4096 @ d32768 | 541.41 ± 0.20 |
| p02 | defaut | q4ks | -fa on -ub 2048 -p 4096 -n 128 -d 32768 | tg128 @ d32768 | 51.57 ± 0.08 |
| p02 | defaut | q4ks | -fa on -ctk q8_0 -ctv q8_0 -ub 2048 -p 4096 -n 128 -d 32768 | pp4096 @ d32768 | 538.74 ± 0.79 |
| p02 | defaut | q4ks | -fa on -ctk q8_0 -ctv q8_0 -ub 2048 -p 4096 -n 128 -d 32768 | tg128 @ d32768 | 43.33 ± 0.19 |
| p03 | defaut | q4ks | -fa off -ub 2048 -p 4096 -n 128 -d 0 | pp4096 | 1628.62 ± 1.97 |
| p03 | defaut | q4ks | -fa off -ub 2048 -p 4096 -n 128 -d 0 | tg128 | 60.31 ± 0.02 |
| p03 | defaut | q4ks | -fa off -ub 2048 -p 4096 -n 128 -d 32768 | échec | llama_bench: error: failed to create context with model '/root/models/q4ks.gguf' |
| p03 | defaut | q4ks | -fa on -ub 2048 -p 4096 -n 128 -d 32768 | pp4096 @ d32768 | 540.83 ± 0.65 |
| p03 | defaut | q4ks | -fa on -ub 2048 -p 4096 -n 128 -d 32768 | tg128 @ d32768 | 51.37 ± 0.11 |
| p03 | defaut | q4ks | -fa on -ctk q8_0 -ctv q8_0 -ub 2048 -p 4096 -n 128 -d 32768 | pp4096 @ d32768 | 539.20 ± 0.25 |
| p03 | defaut | q4ks | -fa on -ctk q8_0 -ctv q8_0 -ub 2048 -p 4096 -n 128 -d 32768 | tg128 @ d32768 | 43.54 ± 0.07 |

## 4. A/B des bascules d'environnement (build p02, tg128 seul)

`GGML_CUDA_PASCAL_MMVQ_MULTI` n'agit qu'à 2-4 colonnes : llama-bench génère toujours à 1 colonne,
son effet se lit dans la section serveur (MTP).

| build | variante | quant | options | test | t/s |
|---|---|---|---|---|---|
| master | defaut | q40 | -fa off -p 0 -n 128 -d 0 | tg128 | 64.07 ± 0.04 |
| p02 | defaut | q40 | -fa off -p 0 -n 128 -d 0 | tg128 | 68.89 ± 0.04 |
| p02 | amont | q40 | -fa off -p 0 -n 128 -d 0 | tg128 | 66.75 ± 0.03 |
| p02 | rows0 | q40 | -fa off -p 0 -n 128 -d 0 | tg128 | 66.86 ± 0.05 |
| p02 | rows1 | q40 | -fa off -p 0 -n 128 -d 0 | tg128 | 67.64 ± 0.04 |
| p02 | rows2 | q40 | -fa off -p 0 -n 128 -d 0 | tg128 | 69.29 ± 0.08 |
| p02 | rows4 | q40 | -fa off -p 0 -n 128 -d 0 | tg128 | 67.15 ± 0.05 |
| p02 | multi0 | q40 | -fa off -p 0 -n 128 -d 0 | tg128 | 68.76 ± 0.08 |
| p02 | q8cache0 | q40 | -fa off -p 0 -n 128 -d 0 | tg128 | 68.58 ± 0.04 |
| p02 | sans-graphs | q40 | -fa off -p 0 -n 128 -d 0 | tg128 | 68.37 ± 0.29 |
| p02 | graph-opt | q40 | -fa off -p 0 -n 128 -d 0 | tg128 | 68.04 ± 0.39 |
| p02 | graph-1bloc | q40 | -fa off -p 0 -n 128 -d 0 | tg128 | 68.34 ± 0.20 |
| p02 | rows4+1bloc | q40 | -fa off -p 0 -n 128 -d 0 | tg128 | 66.91 ± 0.17 |
| p02 | defaut | q40 | -fa off -ub 2048 -p 0 -n 128 -d 32768 | tg128 @ d32768 | 22.91 ± 0.01 |
| p02 | amont | q40 | -fa off -ub 2048 -p 0 -n 128 -d 32768 | tg128 @ d32768 | 22.64 ± 0.02 |
| p02 | rows4 | q40 | -fa off -ub 2048 -p 0 -n 128 -d 32768 | tg128 @ d32768 | 22.71 ± 0.02 |
| p02 | q8cache0 | q40 | -fa off -ub 2048 -p 0 -n 128 -d 32768 | tg128 @ d32768 | 22.88 ± 0.04 |
| p02 | sans-graphs | q40 | -fa off -ub 2048 -p 0 -n 128 -d 32768 | tg128 @ d32768 | 22.87 ± 0.01 |
| master | defaut | q4ks | -fa off -p 0 -n 128 -d 0 | tg128 | 56.09 ± 0.02 |
| p02 | defaut | q4ks | -fa off -p 0 -n 128 -d 0 | tg128 | 60.35 ± 0.09 |
| p02 | amont | q4ks | -fa off -p 0 -n 128 -d 0 | tg128 | 58.24 ± 0.10 |
| p02 | rows0 | q4ks | -fa off -p 0 -n 128 -d 0 | tg128 | 58.22 ± 0.15 |
| p02 | rows1 | q4ks | -fa off -p 0 -n 128 -d 0 | tg128 | 60.48 ± 0.08 |
| p02 | rows2 | q4ks | -fa off -p 0 -n 128 -d 0 | tg128 | 60.94 ± 0.05 |
| p02 | rows4 | q4ks | -fa off -p 0 -n 128 -d 0 | tg128 | 59.05 ± 0.03 |
| p02 | multi0 | q4ks | -fa off -p 0 -n 128 -d 0 | tg128 | 60.24 ± 0.08 |
| p02 | q8cache0 | q4ks | -fa off -p 0 -n 128 -d 0 | tg128 | 60.13 ± 0.08 |
| p02 | sans-graphs | q4ks | -fa off -p 0 -n 128 -d 0 | tg128 | 60.22 ± 0.02 |
| p02 | graph-opt | q4ks | -fa off -p 0 -n 128 -d 0 | tg128 | 60.11 ± 0.07 |
| p02 | graph-1bloc | q4ks | -fa off -p 0 -n 128 -d 0 | tg128 | 60.08 ± 0.10 |
| p02 | rows4+1bloc | q4ks | -fa off -p 0 -n 128 -d 0 | tg128 | 58.80 ± 0.16 |
| p02 | defaut | q4ks | -fa off -ub 2048 -p 0 -n 128 -d 32768 | échec | llama_bench: error: failed to create context with model '/root/models/q4ks.gguf' |
| p02 | amont | q4ks | -fa off -ub 2048 -p 0 -n 128 -d 32768 | échec | llama_bench: error: failed to create context with model '/root/models/q4ks.gguf' |
| p02 | rows4 | q4ks | -fa off -ub 2048 -p 0 -n 128 -d 32768 | échec | llama_bench: error: failed to create context with model '/root/models/q4ks.gguf' |
| p02 | q8cache0 | q4ks | -fa off -ub 2048 -p 0 -n 128 -d 32768 | échec | llama_bench: error: failed to create context with model '/root/models/q4ks.gguf' |
| p02 | sans-graphs | q4ks | -fa off -ub 2048 -p 0 -n 128 -d 32768 | échec | llama_bench: error: failed to create context with model '/root/models/q4ks.gguf' |

## 5. Génération réelle (llama-server, q40, 600 tokens, temp 0, une requête à la fois)

Prompt : texte réel `/root/models/prompt.txt` (≈ 2 000 puis ≈ 30 000 tokens).

| build | variante | spéculatif | prompt | tokens prompt | pp t/s | tokens générés | tg t/s | brouillons acceptés/proposés |
|---|---|---|---|---|---|---|---|---|
| master | defaut | sans | — | serveur non démarré : 0.07.421.760 E srv  server_run_i: exiting due to model loading error | | | | |
| master | defaut | MTP n-max 2 | — | serveur non démarré : 0.07.955.125 E srv  server_run_i: exiting due to model loading error | | | | |
| master | defaut | MTP n-max 3 | — | serveur non démarré : 0.07.424.838 E srv  server_run_i: exiting due to model loading error | | | | |
| p02 | defaut | sans | — | serveur non démarré : 0.06.981.599 E srv  server_run_i: exiting due to model loading error | | | | |
| p02 | defaut | MTP n-max 2 | — | serveur non démarré : 0.07.392.651 E srv  server_run_i: exiting due to model loading error | | | | |
| p02 | defaut | MTP n-max 3 | — | serveur non démarré : 0.07.385.465 E srv  server_run_i: exiting due to model loading error | | | | |
| p03 | defaut | sans | — | serveur non démarré : 0.06.772.813 E srv  server_run_i: exiting due to model loading error | | | | |
| p03 | defaut | MTP n-max 2 | — | serveur non démarré : 0.07.337.494 E srv  server_run_i: exiting due to model loading error | | | | |
| p03 | defaut | MTP n-max 3 | — | serveur non démarré : 0.07.344.099 E srv  server_run_i: exiting due to model loading error | | | | |
| p02 | amont | sans | — | serveur non démarré : 0.06.941.811 E srv  server_run_i: exiting due to model loading error | | | | |
| p02 | rows4 | sans | — | serveur non démarré : 0.06.847.264 E srv  server_run_i: exiting due to model loading error | | | | |
| p02 | multi0 | MTP n-max 2 | — | serveur non démarré : 0.07.538.499 E srv  server_run_i: exiting due to model loading error | | | | |
| p02 | multi0 | MTP n-max 3 | — | serveur non démarré : 0.07.521.429 E srv  server_run_i: exiting due to model loading error | | | | |
| p02 | rows4 | MTP n-max 2 | — | serveur non démarré : 0.07.530.597 E srv  server_run_i: exiting due to model loading error | | | | |

## 6. Coût d'un pas MTP (build p03, llama-server, q40, 600 tokens, temp 0)

Lecture : pas/s = tg / (1 + acceptés/pas). Comparer à la ligne « sans » : un pas MTP n-max 2 coûtait 1,77 pas de base sur master.

| build | variante | spéculatif | prompt | tokens prompt | pp t/s | tokens générés | tg t/s | brouillons acceptés/proposés |
|---|---|---|---|---|---|---|---|---|
| p03 | defaut | sans | — | serveur non démarré : 0.06.862.662 E srv  server_run_i: exiting due to model loading error | | | | |
| p03 | defaut | MTP n-max 2 | — | serveur non démarré : 0.07.709.118 E srv  server_run_i: exiting due to model loading error | | | | |
| p03 | cle-amont | MTP n-max 2 | — | serveur non démarré : 0.07.825.972 E srv  server_run_i: exiting due to model loading error | | | | |
| p03 | sans-graphs | MTP n-max 2 | — | serveur non démarré : 0.08.070.108 E srv  server_run_i: exiting due to model loading error | | | | |
| p03 | tete-150016 | MTP n-max 2 | — | serveur non démarré : 0.07.412.112 E srv  server_run_i: exiting due to model loading error | | | | |
| p03 | tete-100352 | MTP n-max 2 | — | serveur non démarré : 0.08.074.766 E srv  server_run_i: exiting due to model loading error | | | | |
| p03 | tete-65536 | MTP n-max 2 | — | serveur non démarré : 0.07.894.228 E srv  server_run_i: exiting due to model loading error | | | | |
| p03 | tete-32768 | MTP n-max 2 | — | serveur non démarré : 0.07.929.266 E srv  server_run_i: exiting due to model loading error | | | | |
| p03 | tete-100352 | MTP n-max 3 | — | serveur non démarré : 0.07.917.056 E srv  server_run_i: exiting due to model loading error | | | | |
| p03 | defaut | MTP n-max 1 | — | serveur non démarré : 0.07.788.743 E srv  server_run_i: exiting due to model loading error | | | | |
| p03 | pmin-0.5 | MTP n-max 2 p-min 0.5 | — | serveur non démarré : 0.07.759.736 E srv  server_run_i: exiting due to model loading error | | | | |
| p03 | pmin-0.5 | MTP n-max 3 p-min 0.5 | — | serveur non démarré : 0.07.852.033 E srv  server_run_i: exiting due to model loading error | | | | |
| p03 | pmin-0.75 | MTP n-max 2 p-min 0.75 | — | serveur non démarré : 0.07.939.682 E srv  server_run_i: exiting due to model loading error | | | | |
| p03 | pmin-0.75 | MTP n-max 3 p-min 0.75 | — | serveur non démarré : 0.07.991.113 E srv  server_run_i: exiting due to model loading error | | | | |
| p03 | pmin-0.9 | MTP n-max 2 p-min 0.9 | — | serveur non démarré : 0.08.045.177 E srv  server_run_i: exiting due to model loading error | | | | |
| p03 | pmin-0.9 | MTP n-max 3 p-min 0.9 | — | serveur non démarré : 0.07.957.683 E srv  server_run_i: exiting due to model loading error | | | | |
| p03 | tete-100352+pmin-0.75 | MTP n-max 3 p-min 0.75 | — | serveur non démarré : 0.07.770.202 E srv  server_run_i: exiting due to model loading error | | | | |
| p03 | echant-cpu | MTP n-max 2, échantillonnage brouillon CPU | — | serveur non démarré : 0.07.758.715 E srv  server_run_i: exiting due to model loading error | | | | |

_Terminé : 2026-11-11 02:15 UTC. Journaux détaillés dans /root/fork/logs._
