# Fork Pascal (P40, sm_61) — résultats

- Date : 2026-11-11 02:39 UTC
- Commit de base : aa75ff3239f9cc675b081697deda9e1b112345c6
- GPU : Tesla P40, 580.126.09, 24576 MiB
- nvcc : Build cuda_12.4.r12.4/compiler.34097967_0
- Noyau : 5.15.0-171-generic — Ubuntu 22.04.4 LTS
- Builds : master = amont ; p01 = + PR #27721 #25479 #31352 ; p02 = p01 + optimisations du fork ; p03 = p02 + 03-mtp.patch (coût d'un pas MTP)

## 5. Génération réelle (llama-server, q40, 600 tokens, temp 0, une requête à la fois)

Prompt : texte réel `/root/models/prompt.txt` (≈ 2 000 puis ≈ 30 000 tokens).

| build | variante | spéculatif | prompt | tokens prompt | pp t/s | tokens générés | tg t/s | brouillons acceptés/proposés |
|---|---|---|---|---|---|---|---|---|
| master | defaut | sans | court | 2171 | 1373 | 600 | **60.93** | —/— |
| master | defaut | sans | long | 18861 | 1084.9 | 600 | **55.24** | —/— |
| master | defaut | MTP n-max 2 | court | 2171 | 1150.5 | 600 | **79.94** | 349/498 |
| master | defaut | MTP n-max 2 | long | 18861 | 1004.3 | 600 | **62.77** | 309/577 |
| master | defaut | MTP n-max 3 | court | 2171 | 1165.5 | 600 | **80.12** | 392/619 |
| master | defaut | MTP n-max 3 | long | 18861 | 1008.4 | 600 | **62.28** | 352/736 |
| p02 | defaut | sans | court | 2171 | 1573.5 | 600 | **66.07** | —/— |
| p02 | defaut | sans | long | 18861 | 1181.8 | 600 | **60** | —/— |
| p02 | defaut | MTP n-max 2 | court | 2171 | 1264.2 | 600 | **85.55** | 358/482 |
| p02 | defaut | MTP n-max 2 | long | 18861 | 1097.2 | 600 | **70.81** | 331/535 |
| p02 | defaut | MTP n-max 3 | court | 2171 | 1283 | 600 | **81.66** | 398/602 |
| p02 | defaut | MTP n-max 3 | long | 18861 | 1093.3 | 600 | **55.97** | 330/805 |
| p03 | defaut | sans | court | 2171 | 1572 | 600 | **65.52** | —/— |
| p03 | defaut | sans | long | 18861 | 1182 | 600 | **59.23** | —/— |
| p03 | defaut | MTP n-max 2 | court | 2171 | 1285.5 | 600 | **88.1** | 358/482 |
| p03 | defaut | MTP n-max 2 | long | 18861 | 1095.2 | 600 | **70.18** | 331/535 |
| p03 | defaut | MTP n-max 3 | court | 2171 | 1278.5 | 600 | **83.02** | 398/602 |
| p03 | defaut | MTP n-max 3 | long | 18861 | 1093.2 | 600 | **57.34** | 330/805 |
| p02 | amont | sans | court | 2171 | 1571.6 | 600 | **63.8** | —/— |
| p02 | amont | sans | long | 18861 | 1181 | 600 | **57.36** | —/— |
| p02 | rows4 | sans | court | 2171 | 1566.3 | 600 | **64.87** | —/— |
| p02 | rows4 | sans | long | 18861 | 1180.6 | 600 | **59.06** | —/— |
| p02 | multi0 | MTP n-max 2 | court | 2171 | 1282.9 | 600 | **84.29** | 349/498 |
| p02 | multi0 | MTP n-max 2 | long | 18861 | 1095.2 | 600 | **64.56** | 310/575 |
| p02 | multi0 | MTP n-max 3 | court | 2171 | 1246.2 | 600 | **79.76** | 391/622 |
| p02 | multi0 | MTP n-max 3 | long | 18861 | 1094.8 | 600 | **62.29** | 352/736 |
| p02 | rows4 | MTP n-max 2 | court | 2171 | 1286.9 | 600 | **86.45** | 358/482 |
| p02 | rows4 | MTP n-max 2 | long | 18861 | 1096.5 | 600 | **69.01** | 331/535 |

## 6. Coût d'un pas MTP (build p03, llama-server, q40, 600 tokens, temp 0)

Lecture : pas/s = tg / (1 + acceptés/pas). Comparer à la ligne « sans » : un pas MTP n-max 2 coûtait 1,77 pas de base sur master.

| build | variante | spéculatif | prompt | tokens prompt | pp t/s | tokens générés | tg t/s | brouillons acceptés/proposés |
|---|---|---|---|---|---|---|---|---|
| p03 | defaut | sans | court | 2171 | 1523.2 | 600 | **65.52** | —/— |
| p03 | defaut | sans | long | 18861 | 1176.9 | 600 | **59.43** | —/— |
| p03 | defaut | MTP n-max 2 | court | 2171 | 1245.7 | 600 | **85.6** | 358/482 |
| p03 | defaut | MTP n-max 2 | long | 18861 | 1092.4 | 600 | **69.22** | 331/535 |
| p03 | cle-amont | MTP n-max 2 | court | 2171 | 1288.9 | 600 | **86.39** | 358/482 |
| p03 | cle-amont | MTP n-max 2 | long | 18861 | 1096.9 | 600 | **71.97** | 331/535 |
| p03 | sans-graphs | MTP n-max 2 | court | 2171 | 1211.6 | 600 | **84.1** | 358/482 |
| p03 | sans-graphs | MTP n-max 2 | long | 18861 | 1093.1 | 600 | **68.92** | 331/535 |
| p03 | tete-150016 | MTP n-max 2 | court | 2171 | 1216.8 | 600 | **80.05** | 328/540 |
| p03 | tete-150016 | MTP n-max 2 | long | 18861 | 1090.8 | 600 | **66.41** | 305/585 |
| p03 | tete-100352 | MTP n-max 2 | court | 2171 | 1271.2 | 600 | **83.19** | 328/540 |
| p03 | tete-100352 | MTP n-max 2 | long | 18861 | 1094.9 | 600 | **66.23** | 305/585 |
| p03 | tete-65536 | MTP n-max 2 | court | 2171 | 1283.2 | 600 | **79.54** | 315/566 |
| p03 | tete-65536 | MTP n-max 2 | long | 18861 | 1095.8 | 600 | **68.14** | 295/605 |
| p03 | tete-32768 | MTP n-max 2 | court | 2171 | 1276 | 600 | **79.98** | 305/586 |
| p03 | tete-32768 | MTP n-max 2 | long | 18861 | 1098.3 | 600 | **67.08** | 285/625 |
| p03 | tete-100352 | MTP n-max 3 | court | 2171 | 1265.3 | 600 | **77.38** | 365/701 |
| p03 | tete-100352 | MTP n-max 3 | long | 18861 | 1090.5 | 600 | **57.89** | 312/860 |
| p03 | defaut | MTP n-max 1 | court | 2171 | 1276.8 | 600 | **81.19** | 275/324 |
| p03 | defaut | MTP n-max 1 | long | 18861 | 1096.2 | 600 | **71.06** | 258/340 |
| p03 | pmin-0.5 | MTP n-max 2 p-min 0.5 | court | 2171 | 1275.7 | 600 | **84.14** | 345/411 |
| p03 | pmin-0.5 | MTP n-max 2 p-min 0.5 | long | 18861 | 1094.3 | 600 | **69.08** | 303/372 |
| p03 | pmin-0.5 | MTP n-max 3 p-min 0.5 | court | 2171 | 1282.6 | 600 | **84.74** | 387/480 |
| p03 | pmin-0.5 | MTP n-max 3 p-min 0.5 | long | 18861 | 1095.5 | 600 | **65.47** | 318/430 |
| p03 | pmin-0.75 | MTP n-max 2 p-min 0.75 | court | 2171 | 1283.4 | 600 | **80.56** | 319/335 |
| p03 | pmin-0.75 | MTP n-max 2 p-min 0.75 | long | 18861 | 1097.2 | 600 | **64.93** | 258/274 |
| p03 | pmin-0.75 | MTP n-max 3 p-min 0.75 | court | 2171 | 1276.2 | 600 | **74.68** | 342/375 |
| p03 | pmin-0.75 | MTP n-max 3 p-min 0.75 | long | 18861 | 1096 | 600 | **69** | 301/325 |
| p03 | pmin-0.9 | MTP n-max 2 p-min 0.9 | court | 2171 | 1272.1 | 600 | **72.85** | 282/287 |
| p03 | pmin-0.9 | MTP n-max 2 p-min 0.9 | long | 18861 | 1095.2 | 600 | **61.2** | 211/214 |
| p03 | pmin-0.9 | MTP n-max 3 p-min 0.9 | court | 2171 | 1277.8 | 600 | **74.55** | 295/303 |
| p03 | pmin-0.9 | MTP n-max 3 p-min 0.9 | long | 18861 | 1094.9 | 600 | **60.21** | 206/212 |
| p03 | tete-100352+pmin-0.75 | MTP n-max 3 p-min 0.75 | court | 2171 | 1279 | 600 | **76.82** | 314/366 |
| p03 | tete-100352+pmin-0.75 | MTP n-max 3 p-min 0.75 | long | 18861 | 1093.2 | 600 | **67.83** | 269/299 |
| p03 | echant-cpu | MTP n-max 2, échantillonnage brouillon CPU | court | 2171 | 1282 | 600 | **85.68** | 358/482 |
| p03 | echant-cpu | MTP n-max 2, échantillonnage brouillon CPU | long | 18861 | 1095.5 | 600 | **70.04** | 331/535 |

_Terminé : 2026-11-11 03:06 UTC. Journaux détaillés dans /root/fork/logs._
