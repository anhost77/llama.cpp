#!/usr/bin/env bash
# bench.sh — compile 3 builds de llama.cpp (master, master+01, master+01+02) pour Tesla P40 (sm_61),
# vérifie la justesse puis mesure pp/tg (llama-bench) et la génération réelle (llama-server, avec/sans MTP).
# Cible : Ubuntu 22.04 + CUDA 12.4, exécuté en root. Résultats en markdown dans $OUT, écrits au fur et à mesure.
#
# Usage : bash bench.sh            (tout)
#         ETAPES="build justesse" bash bench.sh   (sous-ensemble : build justesse bench ab serveur)
set -u

# ----------------------------------------------------------------------------- réglages
MODELS="${MODELS:-/root/models}"
Q40="${Q40:-$MODELS/q40.gguf}"
Q4KS="${Q4KS:-$MODELS/q4ks.gguf}"
MTP_MODEL="${MTP_MODEL:-$MODELS/mtp-q40.gguf}"      # optionnel : tests MTP sautés si absent
PROMPT_FILE="${PROMPT_FILE:-$MODELS/prompt.txt}"    # optionnel : vrai texte français (chapitres) pour le test serveur
OUT="${OUT:-/root/web/fork.txt}"
WORK="${WORK:-/root/fork}"
PATCHES="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="${REPO_URL:-https://github.com/ggml-org/llama.cpp}"
PORT="${PORT:-8089}"
CTX="${CTX:-40000}"
REPS="${REPS:-3}"
DEPTH="${DEPTH:-32768}"
ETAPES="${ETAPES:-build justesse bench ab serveur mtp}"
BUILDS="${BUILDS:-master p01 p02 p03}"
LOGS="$WORK/logs"

export PATH="/usr/local/cuda/bin:/usr/local/cuda-12.4/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# Variantes d'environnement mesurées en A/B sur le build p02 ("nom|VAR=val VAR=val").
# La première ligne est le défaut du fork ; "amont" doit redonner le comportement de master.
# Défaut du fork : MMVQ_ROWS non défini = auto (4 lignes par bloc si K <= 1024, sinon 2), MULTI=1, Q8_CACHE=1.
AB_VARIANTES=(
  "defaut|"
  "amont|GGML_CUDA_PASCAL_MMVQ_ROWS=0 GGML_CUDA_PASCAL_Q8_CACHE=0 GGML_CUDA_DISABLE_GRAPHS=1"
  "rows0|GGML_CUDA_PASCAL_MMVQ_ROWS=0"
  "rows1|GGML_CUDA_PASCAL_MMVQ_ROWS=1"
  "rows2|GGML_CUDA_PASCAL_MMVQ_ROWS=2"
  "rows4|GGML_CUDA_PASCAL_MMVQ_ROWS=4"
  "multi0|GGML_CUDA_PASCAL_MMVQ_MULTI=0"
  "q8cache0|GGML_CUDA_PASCAL_Q8_CACHE=0"
  "sans-graphs|GGML_CUDA_DISABLE_GRAPHS=1"
  "graph-opt|GGML_CUDA_GRAPH_OPT=1"
  "graph-1bloc|GGML_CUDA_GRAPH_CHUNK_SIZE=0"
  "rows4+1bloc|GGML_CUDA_PASCAL_MMVQ_ROWS=4 GGML_CUDA_GRAPH_CHUNK_SIZE=0"
)

# ----------------------------------------------------------------------------- utilitaires
mkdir -p "$(dirname "$OUT")" "$WORK" "$LOGS"

out()  { printf '%s\n' "$*" >> "$OUT"; }
log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
a_etape() { case " $ETAPES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
bin()  { printf '%s/%s/build/bin/%s' "$WORK" "$1" "$2"; }          # bin <build> <outil>
build_ok() { [ -x "$(bin "$1" llama-bench)" ]; }
nom_var()  { printf '%s' "${1%%|*}"; }
env_var()  { printf '%s' "${1#*|}"; }
# run_env "VAR=a VAR=b" commande... : exécute la commande avec ces variables d'environnement
run_env() { local e="$1"; shift; if [ -n "$e" ]; then env $e "$@"; else "$@"; fi; }

entete() {
  : > "$OUT"
  out "# Fork Pascal (P40, sm_61) — résultats"
  out ""
  out "- Date : $(date -u '+%Y-%m-%d %H:%M UTC')"
  out "- Commit de base : $(cat "$PATCHES/BASE" 2>/dev/null)"
  out "- GPU : $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1)"
  out "- nvcc : $(nvcc --version 2>/dev/null | tail -1)"
  out "- Noyau : $(uname -r) — $(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-?}")"
  out "- Builds : master = amont ; p01 = + PR #27721 #25479 #31352 ; p02 = p01 + optimisations du fork ; p03 = p02 + 03-mtp.patch (coût d'un pas MTP)"
  out ""
}

# ----------------------------------------------------------------------------- dépendances et sources
installer_deps() {
  local manquants="" p
  for p in git cmake g++ make curl jq python3 unzip; do
    command -v "$p" >/dev/null 2>&1 || manquants="$manquants $p"
  done
  if [ -n "$manquants" ] || [ ! -e /usr/include/x86_64-linux-gnu/curl/curl.h ]; then
    log "installation des paquets (manquants :$manquants)"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y > "$LOGS/apt.log" 2>&1
    apt-get install -y git cmake build-essential curl jq python3 unzip libcurl4-openssl-dev libssl-dev >> "$LOGS/apt.log" 2>&1 \
      || out "> ATTENTION : apt-get install a échoué (voir $LOGS/apt.log)"
  fi
}

preparer_sources() {
  local base; base="$(tr -d ' \r\n' < "$PATCHES/BASE")"
  if [ ! -d "$WORK/src/.git" ]; then
    log "clonage de $REPO_URL"
    git clone "$REPO_URL" "$WORK/src" > "$LOGS/clone.log" 2>&1 || { out "> ERREUR : clonage impossible"; return 1; }
  fi
  git -C "$WORK/src" cat-file -e "$base^{commit}" 2>/dev/null || git -C "$WORK/src" fetch origin "$base" >> "$LOGS/clone.log" 2>&1
  local b
  for b in $BUILDS; do
    [ -d "$WORK/$b" ] && continue
    git -C "$WORK/src" worktree add --detach "$WORK/$b" "$base" >> "$LOGS/clone.log" 2>&1 \
      || { out "> ERREUR : worktree $b impossible (commit $base introuvable ?)"; continue; }
    case "$b" in
      p01) appliquer "$b" 01-prs.patch ;;
      p02) appliquer "$b" 01-prs.patch && appliquer "$b" 02-fork.patch ;;
      p03) appliquer "$b" 01-prs.patch && appliquer "$b" 02-fork.patch && appliquer "$b" 03-mtp.patch ;;
    esac
  done
}

appliquer() { # appliquer <build> <patch>
  if git -C "$WORK/$1" apply --whitespace=nowarn "$PATCHES/$2" 2> "$LOGS/apply-$1-$2.log"; then
    return 0
  fi
  out "> ERREUR : \`git apply $2\` a échoué sur $1 :"
  out '```'; tail -20 "$LOGS/apply-$1-$2.log" >> "$OUT"; out '```'
  touch "$WORK/$1/.patch-echec"
  return 1
}

compiler() { # compiler <build>
  local b="$1" d="$WORK/$1"
  [ -d "$d" ] || return 1
  [ -e "$d/.patch-echec" ] && { out "- $b : non compilé (patch non appliqué)"; return 1; }
  build_ok "$b" && [ -x "$(bin "$b" test-backend-ops)" ] && { out "- $b : déjà compilé"; return 0; }
  log "compilation de $b"
  local t0=$SECONDS
  cmake -S "$d" -B "$d/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_BUILD_TESTS=ON > "$LOGS/cmake-$b.log" 2>&1
  cmake --build "$d/build" -j "$(nproc)" \
        --target llama-bench llama-server llama-perplexity llama-completion test-backend-ops > "$LOGS/build-$b.log" 2>&1
  if build_ok "$b"; then
    out "- $b : compilé en $((SECONDS - t0)) s"
  else
    out "- $b : **ÉCHEC de compilation** (fin de $LOGS/build-$b.log) :"
    out '```'; grep -n -i -B2 -A6 'error' "$LOGS/build-$b.log" | head -60 >> "$OUT"; tail -15 "$LOGS/build-$b.log" >> "$OUT"; out '```'
    return 1
  fi
}

etape_build() {
  out "## 1. Compilation"; out ""
  installer_deps
  preparer_sources || return 1
  local b; for b in $BUILDS; do compiler "$b"; done
  out ""
}

# ----------------------------------------------------------------------------- justesse
tbo() { # tbo <build> <nom variante> <env> : test-backend-ops sur MUL_MAT et MUL_MAT_ID
  local b="$1" nom="$2" e="$3" f="$LOGS/tbo-$1-$2.log" exe
  exe="$(bin "$b" test-backend-ops)"
  [ -x "$exe" ] || { out "| $b | $nom | test-backend-ops absent | |"; return; }
  run_env "$e" timeout 3600 "$exe" test -b CUDA0 -o MUL_MAT,MUL_MAT_ID > "$f" 2>&1
  local rc=$? res
  res="$(grep -E '[0-9]+/[0-9]+ tests passed' "$f" | tail -1 | sed 's/^ *//')"
  out "| $b | $nom | ${res:-aucun résultat (voir $f)} | $([ $rc -eq 0 ] && echo OK || echo "**code $rc**") |"
}

PROMPT_DET="Voici le début d'un roman policier se déroulant à Lyon en hiver. Chapitre premier. Le commissaire"

sortie_det() { # sortie_det <build> <nom> <env> <modèle> <étiquette quant> : génération déterministe -> fichier
  local exe; exe="$(bin "$1" llama-completion)"
  local f="$LOGS/det-$1-$2-$5.txt"
  [ -x "$exe" ] || return 1
  run_env "$3" timeout 900 "$exe" -m "$4" -ngl 99 -fa off -c 4096 -no-cnv -n 200 --temp 0 -s 42 \
      --no-display-prompt --simple-io -p "$PROMPT_DET" < /dev/null > "$f" 2> "$LOGS/det-$1-$2-$5.err"
  printf '%s' "$f"
}

comparer_det() { # comparer_det <réf> <fichier> : "identique" ou position de la première différence
  [ -s "$2" ] || { printf 'sortie vide (voir .err)'; return; }
  if cmp -s "$1" "$2"; then printf 'identique'; return; fi
  local octet; octet="$(LC_ALL=C cmp "$1" "$2" 2>/dev/null | sed -n -E 's/.*byte ([0-9]+).*/\1/p' | head -1)"
  printf 'diffère à l'"'"'octet %s sur %s (≈ token %s)' "${octet:-?}" "$(wc -c < "$1")" "$(( ${octet:-0} / 4 ))"
}

texte_ppl() { # prépare un texte pour la perplexité : wikitext-2 si le réseau répond, sinon texte local
  local f="$WORK/ppl.txt"
  [ -s "$f" ] && { printf '%s' "$f"; return; }
  if curl -fsSL --max-time 120 -o "$WORK/wikitext.zip" \
        "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip" 2>/dev/null \
     && unzip -o -q "$WORK/wikitext.zip" -d "$WORK" 2>/dev/null \
     && [ -s "$WORK/wikitext-2-raw/wiki.test.raw" ]; then
    head -c 200000 "$WORK/wikitext-2-raw/wiki.test.raw" > "$f"
  else
    generer_prompt 12000 > "$f"     # repli : texte français synthétique (comparaison relative seulement)
  fi
  printf '%s' "$f"
}

ppl() { # ppl <build> <env> <modèle> : perplexité courte (16 blocs de 512 tokens)
  local exe; exe="$(bin "$1" llama-perplexity)"
  [ -x "$exe" ] || { printf 'n/d'; return; }
  local f; f="$(texte_ppl)"
  run_env "$2" timeout 1800 "$exe" -m "$3" -ngl 99 -fa off -c 512 -b 2048 --chunks 16 -f "$f" 2>&1 \
    | grep -E 'Final estimate' | tail -1 | sed -E 's/.*PPL = ([0-9.]+).*\+\/- ([0-9.]+).*/\1 ± \2/' | grep . || printf 'n/d'
}

etape_justesse() {
  out "## 2. Justesse"; out ""
  out "### 2.1 test-backend-ops (MUL_MAT, MUL_MAT_ID contre le CPU)"; out ""
  out "| build | variante | résultat | code |"; out "|---|---|---|---|"
  local b v
  for b in master p01; do build_ok "$b" && tbo "$b" defaut ""; done
  if build_ok p02; then
    for v in "${AB_VARIANTES[@]}"; do tbo p02 "$(nom_var "$v")" "$(env_var "$v")"; done
  fi
  out ""

  out "### 2.2 Sortie déterministe (temp 0, seed 42, 200 tokens) comparée à master, et perplexité courte"; out ""
  out "Une divergence n'est pas forcément un bug : l'ordre des sommes flottantes change avec les noyaux."
  out "Dans ce cas, la perplexité doit rester égale à celle de master à l'incertitude près."; out ""
  out "| quant | build | variante | sortie vs master | PPL |"; out "|---|---|---|---|---|"
  local q m ref f
  for q in q40 q4ks; do
    [ "$q" = q40 ] && m="$Q40" || m="$Q4KS"
    [ -f "$m" ] || { out "| $q | — | — | modèle absent ($m) | |"; continue; }
    build_ok master || { out "| $q | master | — | build absent, pas de référence | |"; continue; }
    ref="$(sortie_det master defaut "" "$m" "$q")"
    out "| $q | master | defaut | (référence, $(wc -c < "$ref") octets) | $(ppl master "" "$m") |"
    if build_ok p01; then
      f="$(sortie_det p01 defaut "" "$m" "$q")"
      out "| $q | p01 | defaut | $(comparer_det "$ref" "$f") | $(ppl p01 "" "$m") |"
    fi
    if build_ok p02; then
      for v in "${AB_VARIANTES[@]}"; do
        f="$(sortie_det p02 "$(nom_var "$v")" "$(env_var "$v")" "$m" "$q")"
        # perplexité seulement pour les variantes qui changent les noyaux MMVQ (les autres ne changent pas les calculs)
        case "$(nom_var "$v")" in
          defaut|amont|rows1|rows4|q8cache0) out "| $q | p02 | $(nom_var "$v") | $(comparer_det "$ref" "$f") | $(ppl p02 "$(env_var "$v")" "$m") |" ;;
          *)                                 out "| $q | p02 | $(nom_var "$v") | $(comparer_det "$ref" "$f") | — |" ;;
        esac
      done
    fi
  done
  out ""
}

# ----------------------------------------------------------------------------- llama-bench
# lb <build> <variante> <env> <quant> <modèle> <args llama-bench...> : ajoute les lignes de résultat au tableau
lb() {
  local b="$1" nom="$2" e="$3" q="$4" m="$5"; shift 5
  local exe f; exe="$(bin "$b" llama-bench)"
  f="$LOGS/lb-$b-$nom-$q-$(printf '%s' "$*" | tr -c 'A-Za-z0-9' '_').md"
  run_env "$e" timeout 3600 "$exe" -m "$m" -ngl 99 -r "$REPS" -o md "$@" > "$f" 2> "${f%.md}.err"
  # lignes de données du tableau md de llama-bench : on garde les colonnes test et t/s (les deux dernières)
  local n=0 ligne
  while IFS= read -r ligne; do
    n=$((n + 1))
    out "| $b | $nom | $q | $* | $(printf '%s' "$ligne" | awk -F'|' '{gsub(/^ +| +$/,"",$(NF-2)); gsub(/^ +| +$/,"",$(NF-1)); print $(NF-2) " | " $(NF-1)}') |"
  done < <(grep -E '^\|' "$f" | grep -v -E '^\| *(model|-+:?) ' | grep -E '\| *(pp|tg)[0-9]')
  [ $n -eq 0 ] && out "| $b | $nom | $q | $* | échec | $(tail -1 "${f%.md}.err" | cut -c1-80) |"
}

etape_bench() {
  out "## 3. llama-bench (pp4096 / tg128, -ub 2048, $REPS répétitions)"; out ""
  out "| build | variante | quant | options | test | t/s |"; out "|---|---|---|---|---|---|"
  local b q m
  for q in q40 q4ks; do
    [ "$q" = q40 ] && m="$Q40" || m="$Q4KS"
    [ -f "$m" ] || continue
    for b in $BUILDS; do
      build_ok "$b" || continue
      log "llama-bench $b $q"
      lb "$b" defaut "" "$q" "$m" -fa off -ub 2048 -p 4096 -n 128 -d 0
      lb "$b" defaut "" "$q" "$m" -fa off -ub 2048 -p 4096 -n 128 -d "$DEPTH"
      lb "$b" defaut "" "$q" "$m" -fa on  -ub 2048 -p 4096 -n 128 -d "$DEPTH"
      lb "$b" defaut "" "$q" "$m" -fa on  -ctk q8_0 -ctv q8_0 -ub 2048 -p 4096 -n 128 -d "$DEPTH"
    done
  done
  out ""
}

etape_ab() {
  out "## 4. A/B des bascules d'environnement (build p02, tg128 seul)"; out ""
  build_ok p02 || { out "build p02 absent, étape sautée."; out ""; return; }
  out "\`GGML_CUDA_PASCAL_MMVQ_MULTI\` n'agit qu'à 2-4 colonnes : llama-bench génère toujours à 1 colonne,"
  out "son effet se lit dans la section serveur (MTP)."; out ""
  out "| build | variante | quant | options | test | t/s |"; out "|---|---|---|---|---|---|"
  local q m v
  for q in q40 q4ks; do
    [ "$q" = q40 ] && m="$Q40" || m="$Q4KS"
    [ -f "$m" ] || continue
    build_ok master && lb master defaut "" "$q" "$m" -fa off -p 0 -n 128 -d 0
    build_ok p01    && lb p01    defaut "" "$q" "$m" -fa off -p 0 -n 128 -d 0
    for v in "${AB_VARIANTES[@]}"; do
      log "A/B $(nom_var "$v") $q"
      lb p02 "$(nom_var "$v")" "$(env_var "$v")" "$q" "$m" -fa off -p 0 -n 128 -d 0
    done
    # à grande profondeur : seulement les variantes principales (le remplissage du contexte est long)
    for v in "${AB_VARIANTES[@]:0:2}" "${AB_VARIANTES[5]}" "${AB_VARIANTES[7]}" "${AB_VARIANTES[8]}"; do
      lb p02 "$(nom_var "$v")" "$(env_var "$v")" "$q" "$m" -fa off -ub 2048 -p 0 -n 128 -d "$DEPTH"
    done
  done
  out ""
}

# ----------------------------------------------------------------------------- génération réelle (llama-server)
# Texte français embarqué. Répété tel quel, il gonflerait artificiellement le taux d'acceptation du
# décodage spéculatif : les phrases sont donc mélangées (graine fixe). Un vrai texte ($PROMPT_FILE) reste préférable.
generer_prompt() { # generer_prompt <nb de tokens visés> (≈ 4 caractères par token)
  local cible="$1"
  if [ -s "$PROMPT_FILE" ]; then
    head -c $((cible * 4)) "$PROMPT_FILE"
    printf '\n\nChapitre suivant.\n\n'
    return
  fi
  python3 - "$cible" <<'PY'
import random, sys
cible = int(sys.argv[1]) * 4
phrases = """La pluie tombait sur les quais de Saône depuis le milieu de la nuit, et les réverbères dessinaient des cercles jaunes sur les pavés.
Le commissaire Vasseur referma le dossier sans un mot, puis il regarda longuement la photographie posée sur le bureau.
Dans la cuisine, la vieille horloge marquait toujours sept heures moins dix, comme si le temps avait renoncé à avancer.
Marguerite avait gardé toutes les lettres de son frère dans une boîte en fer, rangée derrière les bocaux de confiture.
Au marché de la Croix-Rousse, on parlait à voix basse de l'homme retrouvé près du funiculaire.
Il fallait pourtant bien que quelqu'un ait ouvert la porte du jardin, puisque le chien n'avait pas aboyé.
Le brigadier Lemoine notait tout dans un petit carnet noir dont il ne se séparait jamais, même pour dormir.
Personne ne se souvenait d'avoir vu la voiture grise avant le mardi, et pourtant elle semblait connaître le quartier.
La boulangère affirmait que le jeune homme était passé deux fois, la seconde sans rien acheter.
Sur le pont, le vent emportait les journaux et les chapeaux, et l'on entendait au loin les cloches de Fourvière.
Elle comprit, en voyant la tasse encore tiède, que l'appartement n'était vide que depuis quelques minutes.
Le médecin légiste haussa les épaules : selon lui, la réponse se trouvait dans les chaussures de la victime.
Un train de marchandises traversa la gare de Perrache sans ralentir, et les vitres du café tremblèrent.
Vasseur détestait les coïncidences, surtout lorsqu'elles portaient un nom de famille qu'il connaissait.
La clef trouvée dans la doublure du manteau n'ouvrait aucune des serrures de la maison.
À la préfecture, on voulait un coupable avant dimanche, peu importait lequel.
Le notaire, un homme sec aux lunettes cerclées d'or, prétendait n'avoir jamais reçu le second testament.
Dans la cour, les enfants avaient tracé à la craie une marelle qui s'arrêtait net devant la remise.
Il restait une odeur de tabac blond dans l'escalier, alors que personne dans l'immeuble ne fumait.
La concierge jurait avoir entendu deux voix, dont l'une riait doucement.""".split("\n")
rnd = random.Random(1234)
sortie, taille, chap = [], 0, 1
while taille < cible:
    lot = rnd.sample(phrases, 6)
    par = f"Chapitre {chap}.\n\n" + " ".join(lot) + "\n\n"
    sortie.append(par); taille += len(par); chap += 1
sys.stdout.write("".join(sortie) + f"Chapitre {chap}.\n\n")
PY
}

SRV_PID=""
arreter_serveur() {
  [ -n "$SRV_PID" ] || return 0
  kill "$SRV_PID" 2>/dev/null
  local i; for i in $(seq 1 30); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$SRV_PID" 2>/dev/null
  wait "$SRV_PID" 2>/dev/null
  # le & lance un sous-shell : tuer aussi le vrai llama-server, et attendre que le port soit libre
  pkill -f "llama-server.*--port $PORT" 2>/dev/null
  for i in $(seq 1 30); do pgrep -f "llama-server.*--port $PORT" >/dev/null || break; sleep 1; done
  pkill -9 -f "llama-server.*--port $PORT" 2>/dev/null; sleep 2
  SRV_PID=""
}
trap arreter_serveur EXIT

demarrer_serveur() { # demarrer_serveur <build> <env> <journal> <args supplémentaires...> ; 0 si /health répond
  local b="$1" e="$2" journal="$3"; shift 3
  curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "port $PORT déjà occupé" > "$journal"; return 1; }
  run_env "$e" "$(bin "$b" llama-server)" -m "$Q40" -ngl 99 -c "$CTX" -np 1 -ub 2048 -b 2048 -fa on \
      --host 127.0.0.1 --port "$PORT" --no-webui "$@" > "$journal" 2>&1 &
  SRV_PID=$!
  local i
  for i in $(seq 1 300); do
    kill -0 "$SRV_PID" 2>/dev/null || return 1
    curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && return 0
    sleep 2
  done
  return 1
}

requete() { # requete <fichier prompt> -> "prompt_n | pp t/s | predicted_n | tg t/s | brouillons acceptés/proposés"
  local corps="$WORK/req.json" rep="$WORK/rep.json"
  jq -n --rawfile p "$1" '{prompt: $p, n_predict: 600, temperature: 0, seed: 42, cache_prompt: false}' > "$corps"
  curl -sS --max-time 1800 -H 'Content-Type: application/json' --data-binary "@$corps" \
       "http://127.0.0.1:$PORT/completion" > "$rep" 2>/dev/null
  jq -r 'if .timings then
           "\(.timings.prompt_n) | \(.timings.prompt_per_second * 10 | round / 10) | \(.timings.predicted_n) | **\(.timings.predicted_per_second * 100 | round / 100)** | \(.timings.draft_n_accepted // "—")/\(.timings.draft_n // "—")"
         else "erreur | \(.error.message // "réponse invalide") | | | " end' "$rep" 2>/dev/null \
    || printf 'erreur | réponse illisible | | | '
}

essai_serveur() { # essai_serveur <build> <nom variante> <env> <étiquette spéculatif> <args serveur...>
  local b="$1" nom="$2" e="$3" spec="$4"; shift 4
  local journal="$LOGS/srv-$b-$nom-$(printf '%s' "$spec" | tr -c 'A-Za-z0-9' '_').log"
  log "serveur $b $nom $spec"
  if demarrer_serveur "$b" "$e" "$journal" "$@"; then
    out "| $b | $nom | $spec | court | $(requete "$WORK/prompt-court.txt") |"
    out "| $b | $nom | $spec | long | $(requete "$WORK/prompt-long.txt") |"
  else
    out "| $b | $nom | $spec | — | serveur non démarré : $(grep -i -E 'error|failed' "$journal" | tail -1 | cut -c1-100) | | | | |"
  fi
  arreter_serveur
}

etape_serveur() {
  out "## 5. Génération réelle (llama-server, q40, 600 tokens, temp 0, une requête à la fois)"; out ""
  [ -f "$Q40" ] || { out "modèle $Q40 absent, étape sautée."; out ""; return; }
  generer_prompt 2000  > "$WORK/prompt-court.txt"
  generer_prompt 30000 > "$WORK/prompt-long.txt"
  if [ -s "$PROMPT_FILE" ]; then out "Prompt : texte réel \`$PROMPT_FILE\` (≈ 2 000 puis ≈ 30 000 tokens)."
  else out "Prompt : texte français **synthétique** (phrases mélangées). Le taux d'acceptation MTP y est probablement"
       out "plus élevé que sur un vrai chapitre : fournir un vrai texte dans \`$PROMPT_FILE\` pour un chiffre réaliste."; fi
  local mtp=1
  [ -f "$MTP_MODEL" ] || { mtp=0; out ""; out "Fichier MTP absent (\`$MTP_MODEL\`) : essais MTP sautés."; }
  out ""
  out "| build | variante | spéculatif | prompt | tokens prompt | pp t/s | tokens générés | tg t/s | brouillons acceptés/proposés |"
  out "|---|---|---|---|---|---|---|---|---|"
  local b n
  for b in $BUILDS; do
    build_ok "$b" && [ -x "$(bin "$b" llama-server)" ] || continue
    essai_serveur "$b" defaut "" "sans"
    if [ $mtp -eq 1 ]; then
      for n in 2 3; do
        essai_serveur "$b" defaut "" "MTP n-max $n" -md "$MTP_MODEL" --spec-type draft-mtp --spec-draft-n-max "$n"
      done
    fi
  done
  if build_ok p02; then
    essai_serveur p02 amont "GGML_CUDA_PASCAL_MMVQ_ROWS=0 GGML_CUDA_PASCAL_Q8_CACHE=0 GGML_CUDA_DISABLE_GRAPHS=1" "sans"
    essai_serveur p02 rows4 "GGML_CUDA_PASCAL_MMVQ_ROWS=4" "sans"
    if [ $mtp -eq 1 ]; then
      essai_serveur p02 multi0 "GGML_CUDA_PASCAL_MMVQ_MULTI=0" "MTP n-max 2" -md "$MTP_MODEL" --spec-type draft-mtp --spec-draft-n-max 2
      essai_serveur p02 multi0 "GGML_CUDA_PASCAL_MMVQ_MULTI=0" "MTP n-max 3" -md "$MTP_MODEL" --spec-type draft-mtp --spec-draft-n-max 3
      essai_serveur p02 rows4  "GGML_CUDA_PASCAL_MMVQ_ROWS=4"  "MTP n-max 2" -md "$MTP_MODEL" --spec-type draft-mtp --spec-draft-n-max 2
    fi
  fi
  out ""
}

# ----------------------------------------------------------------------------- coût d'un pas MTP (build p03)
# Bascules de 03-mtp.patch :
#   LLAMA_MTP_HEAD_ROWS=N        la tête de sortie du brouillon MTP ne calcule que les N premiers tokens du vocabulaire
#                                (non défini ou 0 = tout le vocabulaire, comportement amont). Ne change jamais le texte
#                                généré : un brouillon faux est seulement rejeté. Surveiller la colonne acceptés/proposés.
#   GGML_CUDA_GRAPH_KEY_SHAPE=0  redonne la clé amont des CUDA graphs (défaut 1 : un graphe par taille de batch).
# Options amont croisées : --spec-draft-n-max, --spec-draft-p-min, --no-spec-draft-backend-sampling.
etape_mtp() {
  out "## 6. Coût d'un pas MTP (build p03, llama-server, q40, 600 tokens, temp 0)"; out ""
  build_ok p03 && [ -x "$(bin p03 llama-server)" ] || { out "build p03 absent, étape sautée."; out ""; return; }
  [ -f "$Q40" ] || { out "modèle $Q40 absent, étape sautée."; out ""; return; }
  [ -f "$MTP_MODEL" ] || { out "Fichier MTP absent (\`$MTP_MODEL\`), étape sautée."; out ""; return; }
  [ -s "$WORK/prompt-court.txt" ] || generer_prompt 2000  > "$WORK/prompt-court.txt"
  [ -s "$WORK/prompt-long.txt" ]  || generer_prompt 30000 > "$WORK/prompt-long.txt"
  out "Lecture : pas/s = tg / (1 + acceptés/pas). Comparer à la ligne « sans » : un pas MTP n-max 2 coûtait 1,77 pas de base sur master."
  out ""
  out "| build | variante | spéculatif | prompt | tokens prompt | pp t/s | tokens générés | tg t/s | brouillons acceptés/proposés |"
  out "|---|---|---|---|---|---|---|---|---|"
  local M2=(-md "$MTP_MODEL" --spec-type draft-mtp --spec-draft-n-max 2)
  essai_serveur p03 defaut "" "sans"
  essai_serveur p03 defaut "" "MTP n-max 2" "${M2[@]}"
  essai_serveur p03 cle-amont "GGML_CUDA_GRAPH_KEY_SHAPE=0" "MTP n-max 2" "${M2[@]}"
  essai_serveur p03 sans-graphs "GGML_CUDA_DISABLE_GRAPHS=1" "MTP n-max 2" "${M2[@]}"
  local n
  for n in 150016 100352 65536 32768; do
    essai_serveur p03 "tete-$n" "LLAMA_MTP_HEAD_ROWS=$n" "MTP n-max 2" "${M2[@]}"
  done
  essai_serveur p03 tete-100352 "LLAMA_MTP_HEAD_ROWS=100352" "MTP n-max 3" -md "$MTP_MODEL" --spec-type draft-mtp --spec-draft-n-max 3
  essai_serveur p03 defaut "" "MTP n-max 1" -md "$MTP_MODEL" --spec-type draft-mtp --spec-draft-n-max 1
  local p
  for p in 0.5 0.75 0.9; do
    essai_serveur p03 "pmin-$p" "" "MTP n-max 2 p-min $p" "${M2[@]}" --spec-draft-p-min "$p"
    essai_serveur p03 "pmin-$p" "" "MTP n-max 3 p-min $p" -md "$MTP_MODEL" --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min "$p"
  done
  essai_serveur p03 "tete-100352+pmin-0.75" "LLAMA_MTP_HEAD_ROWS=100352" "MTP n-max 3 p-min 0.75" \
      -md "$MTP_MODEL" --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75
  essai_serveur p03 echant-cpu "" "MTP n-max 2, échantillonnage brouillon CPU" "${M2[@]}" --no-spec-draft-backend-sampling
  out ""
}

# ----------------------------------------------------------------------------- déroulé
entete
a_etape build    && etape_build
a_etape justesse && etape_justesse
a_etape bench    && etape_bench
a_etape ab       && etape_ab
a_etape serveur  && etape_serveur
a_etape mtp      && etape_mtp
out "_Terminé : $(date -u '+%Y-%m-%d %H:%M UTC'). Journaux détaillés dans $LOGS._"
log "terminé, résultats dans $OUT"
