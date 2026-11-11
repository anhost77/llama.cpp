# Séance 05 — suite : recherche de n-grammes dans le prompt + MTP, requêtes de vérification réelles

Requêtes : 8 requêtes réelles (`/root/models/verif-reelles.jsonl`), /v1/chat/completions, temp 0, telles quelles (+ cache_prompt false).
Build 05 par défaut + liste de tokens. t/s = somme des tokens générés / somme des temps de génération.
« textes identiques » : sorties identiques à celles de la première ligne (référence).

| variante | réponses | tokens générés | tg t/s (somme) | brouillons acceptés/proposés | textes identiques | t/s par requête |
|---|---|---|---|---|---|---|
| MTP2 seul (référence) | 8/8 | 2141 | **101.4** | 1306/1656 | 8/8 | 99 100 99 98 105 101 104 100 |
| ngram-simple (n=12, m=48) + MTP2 | 8/8 | 2174 | **75.0** | 1454/3437 | 2/8 | 99 67 72 73 67 77 80 70 |
| ngram-simple n=6 m=32 seul (sans MTP) | 8/8 | 2156 | **65.2** | 928/3673 | 4/8 | 64 54 68 59 69 63 81 59 |
| ngram-simple n=6 m=32 + MTP2 | 8/8 | 1946 | **76.5** | 1373/3272 | 1/8 | 77 71 88 65 84 68 92 59 |
| ngram-mod (défauts) + MTP2 | 8/8 | 2283 | **90.2** | 1438/2799 | 4/8 | 90 100 90 93 75 101 96 73 |
| ngram-simple n=4 m=16 + MTP2 | 8/8 | 2145 | **69.2** | 1469/3568 | 1/8 | 68 58 69 63 73 70 85 60 |
| ngram-map-k (défauts) + MTP2 | 8/8 | 2146 | **96.6** | 1328/1930 | 7/8 | 101 100 100 93 103 98 86 92 |
| MTP2 seul (2e passage) | 8/8 | 2141 | **100.5** | 1306/1656 | 8/8 | 99 100 99 98 100 100 104 101 |

_Terminé : 10:44 UTC, 428 s._
# Séance 05 — suite : recherche de n-grammes dans le prompt + MTP, requêtes de vérification réelles

Requêtes : 8 requêtes réelles (`/root/models/verif-reelles.jsonl`), /v1/chat/completions, temp 0, telles quelles (+ cache_prompt false).
Build 05 par défaut + liste de tokens. t/s = somme des tokens générés / somme des temps de génération.
« textes identiques » : sorties identiques à celles de la première ligne (référence).

| variante | réponses | tokens générés | tg t/s (somme) | brouillons acceptés/proposés | textes identiques | t/s par requête |
|---|---|---|---|---|---|---|
| MTP n-max 3 | 8/8 | 1977 | **103.3** | 1359/1830 | 1/8 | 97 99 104 103 108 104 107 92 |
| MTP n-max 4 | 8/8 | 2103 | **103.0** | 1528/2268 | 3/8 | 102 100 102 98 111 98 109 99 |
| ngram-simple n=8 m=6 + MTP2 | 8/8 | 2141 | **99.8** | 1306/1656 | 8/8 | 98 101 90 99 106 102 104 101 |
| ngram-map-k n=8 m=8 + MTP2 | 8/8 | 2123 | **93.8** | 1297/1783 | 3/8 | 81 97 95 92 98 93 99 86 |
| MTP n-max 3 p-min 0.5 | 8/8 | 2009 | **97.5** | 1357/1625 | 3/8 | 97 104 87 98 103 95 102 99 |
| sans spéculatif | 8/8 | 2141 | **67.5** | 0/0 | 8/8 | 67 67 68 67 67 66 68 68 |

_Terminé : 10:50 UTC._
