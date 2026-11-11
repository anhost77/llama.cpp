- GPU : Tesla P40, sm_61, 30 SM, lancement coopératif : oui
- mode B : 480 blocs de 64 threads (16 par SM), 8 lignes par prise ; médiane de 11 passages

| scénario | étapes | Mo lus | A : µs/étape | A : Go/s | B : µs/étape | B : Go/s | A/B | C : µs/étape | C : Go/s | A/C | résultats |
- mode C : 30 blocs de 1024 threads (1 par SM), partage statique, un atomique par bloc et par étape

|---|---|---|---|---|---|---|---|---|---|---|---
| qkv Q4_0 8192x2048 (9,4 Mo) x40 | 40 | 377.5 | 117.0 | 81 | 153.0 | 62 | 0.76 | 117.7 | 80 | 0.99 | identiques |
| z Q4_0 4096x2048 (4,7 Mo) x40 | 40 | 188.7 | 54.7 | 86 | 67.5 | 70 | 0.81 | 55.3 | 85 | 0.99 | identiques |
| descente experts 16384x512 (5,2 Mo) x40 | 40 | 209.7 | 96.1 | 55 | 132.0 | 40 | 0.73 | 97.3 | 54 | 0.99 | identiques |
| routeur F32 256x2048 (2,1 Mo) x40 | 40 | 83.9 | 14.1 | 149 | 254.4 | 8 | 0.06 | 12.2 | 172 | 1.16 | identiques |
| petit 512x2048 Q4_0 (0,6 Mo) x40 | 40 | 23.6 | 8.5 | 70 | 53.2 | 11 | 0.16 | 9.4 | 63 | 0.91 | identiques |
| portes 64x2048 F32 (0,5 Mo) x40 | 40 | 21.0 | 10.2 | 51 | 166.5 | 3 | 0.06 | 7.0 | 75 | 1.46 | identiques |
| minuscule 2 lignes (8 Ko) x200 | 200 | 1.6 | 4.2 | 2 | 18.4 | 0 | 0.23 | 4.0 | 2 | 1.04 | identiques |
| tête Q6_K 248320x2048 (417 Mo) x2 | 2 | 834.4 | 3098.5 | 135 | 3122.3 | 134 | 0.99 | 3101.7 | 134 | 1.00 | identiques |
| qkv Q4_0 8192x2048 x40, 3 colonnes | 40 | 377.5 | 104.4 | 90 | 190.5 | 50 | 0.55 | 101.6 | 93 | 1.03 | identiques |
| token synthétique (40 couches + tête) | 962 | 2254.2 | 26.7 | 88 | 62.4 | 38 | 0.43 | 25.9 | 90 | 1.03 | identiques |
| token synthétique sans la tête | 961 | 1837.0 | 23.5 | 81 | 61.5 | 31 | 0.38 | 22.7 | 84 | 1.03 | identiques |
| vérification MTP synthétique, 3 colonnes | 1122 | 3428.6 | 49.4 | 62 | 126.4 | 24 | 0.39 | 48.6 | 63 | 1.02 | identiques |

Lecture : A/B > 1 = le noyau persistant est plus rapide. « Go/s » = octets de poids lus / temps total.
