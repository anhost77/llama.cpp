#!/usr/bin/env bash
# session05.sh — séance courte sur une P40 louée : un seul build (BASE + 01 + 02 + 04 + 05), contrôle de justesse,
# profil, puis génération réelle avec et sans les deux fusions de 05.
# Cible : Ubuntu 22.04 + CUDA 12.4 (image nvidia/cuda:12.4.1-devel-ubuntu22.04), exécuté en root.
#
# Usage : BUDGET=2400 bash session05.sh [liste de tokens]      (défaut : $MODELS/tokens2.txt)
# Attendu : $MODELS/q40.gguf, $MODELS/mtp-q40.gguf, $MODELS/prompt.txt, la liste de tokens,
#           et à côté de ce script : BASE, 01-prs.patch, 02-fork.patch, 04-mtp-liste.patch, 05-fusion.patch
#
# La référence est mesurée sur le même build avec les deux fusions coupées (mêmes chemins de code que 04) :
#   LLAMA_PASCAL_GDN_QK_NORM=0 GGML_CUDA_PASCAL_FUSE_GDN_GATES=0
set -u

MODELS="${MODELS:-/root/models}"
Q40="${Q40:-$MODELS/q40.gguf}"
MTP_MODEL="${MTP_MODEL:-$MODELS/mtp-q40.gguf}"
PROMPT_FILE="${PROMPT_FILE:-$MODELS/prompt.txt}"
TOKENS="${1:-${TOKENS:-$MODELS/tokens2.txt}}"
OUT="${OUT:-/root/web/fork5.txt}"
WORK="${WORK:-/root/fork5}"
PATCHES="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="${REPO_URL:-https://github.com/ggml-org/llama.cpp}"
PORT="${PORT:-8089}"
CTX="${CTX:-40000}"
BUDGET="${BUDGET:-2400}"
[ "$BUDGET" -gt 2400 ] && BUDGET=2400
LOGS="$WORK/logs"
SRC="$WORK/p05"
BIN="$SRC/build/bin"
T_DEBUT=$SECONDS
REF="LLAMA_PASCAL_GDN_QK_NORM=0 GGML_CUDA_PASCAL_FUSE_GDN_GATES=0"

export PATH="/usr/local/cuda/bin:/usr/local/cuda-12.4/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "$LOGS" "$(dirname "$OUT")"
out() { printf '%s\n' "$*" >> "$OUT"; }
log() { printf '[%s +%ss] %s\n' "$(date +%H:%M:%S)" "$((SECONDS - T_DEBUT))" "$*" >&2; }
reste() { [ $((SECONDS - T_DEBUT)) -lt "$BUDGET" ]; }
run_env() { local e="$1"; shift; if [ -n "$e" ]; then env $e "$@"; else "$@"; fi; }

: > "$OUT"
out "# Fork Pascal — séance 05 (fusion des petits noyaux du delta-net)"
out ""
out "- Date : $(date -u '+%Y-%m-%d %H:%M UTC')"
out "- GPU : $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1)"
out "- Base : $(tr -d ' \r\n' < "$PATCHES/BASE") + 01 + 02 + 04 + 05 ; liste de tokens : $TOKENS"
out "- Référence = même build, fusions coupées : \`$REF\`"
out ""

# ----------------------------------------------------------------------------- compilation
compiler() {
  out "## 1. Compilation"; out ""
  export DEBIAN_FRONTEND=noninteractive
  local manquants="" p
  for p in git cmake g++ make curl jq python3 unzip; do command -v "$p" >/dev/null 2>&1 || manquants="$manquants $p"; done
  if [ -n "$manquants" ] || [ ! -e /usr/include/x86_64-linux-gnu/curl/curl.h ]; then
    log "installation des paquets (manquants :$manquants)"
    apt-get update -y > "$LOGS/apt.log" 2>&1
    apt-get install -y git cmake build-essential curl jq python3 unzip libcurl4-openssl-dev libssl-dev >> "$LOGS/apt.log" 2>&1
  fi
  local base; base="$(tr -d ' \r\n' < "$PATCHES/BASE")"
  if [ -x "$BIN/llama-server" ] && [ -x "$BIN/llama-perplexity" ]; then out "- déjà compilé"; out ""; return 0; fi
  if [ ! -d "$SRC/.git" ]; then
    log "clonage"
    git clone "$REPO_URL" "$SRC" > "$LOGS/clone.log" 2>&1 || { out "> ERREUR : clonage impossible"; return 1; }
  fi
  git -C "$SRC" cat-file -e "$base^{commit}" 2>/dev/null || git -C "$SRC" fetch origin "$base" >> "$LOGS/clone.log" 2>&1
  git -C "$SRC" checkout -q -f "$base" >> "$LOGS/clone.log" 2>&1 || { out "> ERREUR : commit $base introuvable"; return 1; }
  git -C "$SRC" reset -q --hard "$base"
  local pch
  for pch in 01-prs.patch 02-fork.patch 04-mtp-liste.patch 05-fusion.patch; do
    if ! git -C "$SRC" apply --whitespace=nowarn "$PATCHES/$pch" 2> "$LOGS/apply-$pch.log"; then
      out "> ERREUR : \`git apply $pch\` a échoué :"; out '```'; tail -20 "$LOGS/apply-$pch.log" >> "$OUT"; out '```'
      return 1
    fi
  done
  log "compilation"
  local t0=$SECONDS
  cmake -S "$SRC" -B "$SRC/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_BUILD_TESTS=ON > "$LOGS/cmake.log" 2>&1
  cmake --build "$SRC/build" -j "$(nproc)" \
        --target llama-bench llama-server llama-perplexity llama-completion test-backend-ops > "$LOGS/build.log" 2>&1
  if [ -x "$BIN/llama-server" ] && [ -x "$BIN/llama-bench" ]; then
    out "- compilé en $((SECONDS - t0)) s"
  else
    out "- **ÉCHEC de compilation** :"
    out '```'; grep -n -i -B2 -A6 'error' "$LOGS/build.log" | head -80 >> "$OUT"; tail -15 "$LOGS/build.log" >> "$OUT"; out '```'
    return 1
  fi
  out ""
}

# ----------------------------------------------------------------------------- serveur
SRV_PID=""
arreter_serveur() {
  if [ -n "$SRV_PID" ]; then
    kill "$SRV_PID" 2>/dev/null
    wait "$SRV_PID" 2>/dev/null
  fi
  # le & lance un sous-shell : arrêter aussi le vrai llama-server ([l] : le motif ne se reconnaît pas lui-même)
  pkill -f "[l]lama-server.*--port $PORT" 2>/dev/null
  local i; for i in $(seq 1 30); do pgrep -f "[l]lama-server.*--port $PORT" >/dev/null || break; sleep 1; done
  pkill -9 -f "[l]lama-server.*--port $PORT" 2>/dev/null
  for i in $(seq 1 15); do curl -fsS --max-time 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || break; sleep 1; done
  SRV_PID=""
}
trap arreter_serveur EXIT

demarrer_serveur() { # demarrer_serveur <env> <journal> <args...>
  local e="$1" journal="$2"; shift 2
  curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "port $PORT déjà occupé" > "$journal"; return 1; }
  run_env "$e" "$BIN/llama-server" -m "$Q40" -ngl 99 -c "$CTX" -np 1 -ub 2048 -b 2048 -fa on \
      --host 127.0.0.1 --port "$PORT" --no-webui "$@" > "$journal" 2>&1 &
  SRV_PID=$!
  local i
  for i in $(seq 1 240); do
    kill -0 "$SRV_PID" 2>/dev/null || return 1
    curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && return 0
    sleep 2
  done
  return 1
}

requete() { # requete <fichier prompt> -> colonnes du tableau
  local corps="$WORK/req.json" rep="$WORK/rep.json"
  jq -n --rawfile p "$1" '{prompt: $p, n_predict: 600, temperature: 0, seed: 42, cache_prompt: false}' > "$corps"
  curl -sS --max-time 900 -H 'Content-Type: application/json' --data-binary "@$corps" \
       "http://127.0.0.1:$PORT/completion" > "$rep" 2>/dev/null
  jq -r 'if .timings then
           "\(.timings.prompt_n) | \(.timings.predicted_n) | **\(.timings.predicted_per_second * 100 | round / 100)** | \(.timings.draft_n_accepted // "—")/\(.timings.draft_n // "—")"
         else "erreur | \(.error.message // "réponse invalide") | | " end' "$rep" 2>/dev/null \
    || printf 'erreur | réponse illisible | | '
}

essai() { # essai <nom> <env> <args serveur...> : prompt court puis long
  local nom="$1" e="$2"; shift 2
  reste || { out "| $nom | — | budget de temps épuisé, essai sauté | | | |"; return; }
  local journal="$LOGS/srv-$(printf '%s' "$nom" | tr -c 'A-Za-z0-9' '_').log"
  log "essai $nom"
  if demarrer_serveur "$e" "$journal" "$@"; then
    out "| $nom | court | $(requete "$WORK/prompt-court.txt") |"
    out "| $nom | long | $(requete "$WORK/prompt-long.txt") |"
  else
    out "| $nom | — | serveur non démarré : $(grep -i -E 'error|failed' "$journal" | tail -1 | cut -c1-120) | | | |"
  fi
  arreter_serveur
}

profil() { # profil <nom> <env> : llama-bench sous GGML_CUDA_PROFILE, 25 premières lignes
  local f="$LOGS/profil-$1"
  run_env "$2 GGML_CUDA_PROFILE=1 GGML_CUDA_DISABLE_GRAPHS=1" timeout 200 "$BIN/llama-bench" -m "$Q40" -ngl 99 -fa on -p 0 -n 96 -r 1 \
      > "$f.out" 2> "$f.err"
  out "### $1"; out ""; out '```'
  grep -F '[ggml-cuda profile]' "$f.err" | head -27 >> "$OUT"
  out "nœuds par token (somme des appels / 97 graphes) : $(grep -F '[ggml-cuda profile]' "$f.err" | awk 'NR > 2 { n += $5 } END { printf "%.0f", n / 97 }')"
  out '```'; out ""
}

# ----------------------------------------------------------------------------- déroulé
compiler || { out "_Arrêt : pas de build._"; exit 1; }
[ -f "$Q40" ] || { out "modèle $Q40 absent."; exit 1; }
[ -s "$PROMPT_FILE" ] || { out "prompt $PROMPT_FILE absent."; exit 1; }
{ head -c 8000   "$PROMPT_FILE"; printf '\n\nChapitre suivant.\n\n'; } > "$WORK/prompt-court.txt"
{ head -c 120000 "$PROMPT_FILE"; printf '\n\nChapitre suivant.\n\n'; } > "$WORK/prompt-long.txt"

# --- 2. ordre des nœuds (pour les fusions suivantes) : graphe d'un token, avec et sans la norme q/k commune
out "## 2. Liste des nœuds d'un graphe de génération (1 token)"; out ""
for v in ref 05; do
  [ "$v" = ref ] && e="$REF" || e=""
  run_env "$e GGML_CUDA_PASCAL_DUMP_GRAPH=6 GGML_CUDA_PASCAL_DUMP_FILE=$(dirname "$OUT")/graphe05-$v.txt" \
      timeout 200 "$BIN/llama-bench" -m "$Q40" -ngl 99 -fa on -p 0 -n 12 -r 1 > "$LOGS/dump-$v.out" 2> "$LOGS/dump-$v.err"
  f="$(dirname "$OUT")/graphe05-$v.txt"
  out "- $v : $( [ -s "$f" ] && echo "$(wc -l < "$f") nœuds, fichier $f" || echo "fichier non produit" )"
done
out ""

# --- 3. justesse
out "## 3. Justesse"; out ""
if [ -x "$BIN/test-backend-ops" ]; then
  timeout 900 "$BIN/test-backend-ops" test -b CUDA0 -o RMS_NORM,SCALE,GATED_DELTA_NET > "$LOGS/tbo.log" 2>&1
  rc_tbo=$?
  out "- test-backend-ops RMS_NORM, SCALE, GATED_DELTA_NET : $(grep -E '[0-9]+/[0-9]+ tests passed' "$LOGS/tbo.log" | tail -1 | sed 's/^ *//') (code $rc_tbo)"
fi
PPL_TXT="$WORK/ppl.txt"
if [ ! -s "$PPL_TXT" ]; then
  if curl -fsSL --max-time 120 -o "$WORK/wikitext.zip" "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip" 2>/dev/null \
     && unzip -o -q "$WORK/wikitext.zip" -d "$WORK" 2>/dev/null && [ -s "$WORK/wikitext-2-raw/wiki.test.raw" ]; then
    head -c 200000 "$WORK/wikitext-2-raw/wiki.test.raw" > "$PPL_TXT"
  fi
fi
ppl() { # ppl <env> : même commande que bench.sh (16 blocs de 512 tokens ; master = 7.1616 ± 0.28264 sur wikitext-2)
  run_env "$1" timeout 900 "$BIN/llama-perplexity" -m "$Q40" -ngl 99 -fa off -c 512 -b 2048 --chunks 16 -f "$PPL_TXT" 2>&1 \
    | grep -E 'Final estimate' | tail -1 | sed -E 's/.*PPL = ([0-9.]+).*\+\/- ([0-9.]+).*/\1 ± \2/' | grep . || printf 'n/d'
}
if [ -s "$PPL_TXT" ] && [ -x "$BIN/llama-perplexity" ]; then
  out "- perplexité (master en séance 1 : 7.1616 ± 0.28264) : référence $(ppl "$REF") ; 05 $(ppl "")"
  out "  (la perplexité évalue des lots de 512 tokens : elle couvre la norme q/k commune, pas la fusion des portes, limitée à 64 tokens)"
else
  out "- perplexité : wikitext-2 indisponible, contrôle sauté"
fi
# sortie déterministe 1 token à la fois : couvre la fusion des portes
det() { # det <nom> <env>
  run_env "$2" timeout 300 "$BIN/llama-completion" -m "$Q40" -ngl 99 -fa on -c 4096 -no-cnv -n 200 --temp 0 -s 42 \
      --no-display-prompt --simple-io -p "Voici le début d'un roman policier se déroulant à Lyon en hiver. Chapitre premier. Le commissaire" \
      < /dev/null > "$LOGS/det-$1.txt" 2> "$LOGS/det-$1.err"
}
if [ -x "$BIN/llama-completion" ]; then
  det ref "$REF"; det qk "GGML_CUDA_PASCAL_FUSE_GDN_GATES=0"; det portes "LLAMA_PASCAL_GDN_QK_NORM=0"; det tout ""
  for v in qk portes tout; do
    if cmp -s "$LOGS/det-ref.txt" "$LOGS/det-$v.txt"; then r="identique à la référence ($(wc -c < "$LOGS/det-ref.txt") octets)"
    else r="diffère de la référence à l'octet $(cmp -l "$LOGS/det-ref.txt" "$LOGS/det-$v.txt" 2>/dev/null | head -1 | awk '{print $1}') sur $(wc -c < "$LOGS/det-ref.txt")"; fi
    out "- sortie déterministe (200 tokens, temp 0), variante $v : $r"
  done
  out "  (la norme q/k commune doit être identique au bit près ; la fusion des portes change l'ordre des sommes, une divergence tardive est possible)"
fi
out ""

# --- 4. profil
out "## 4. Profil par type de nœud (llama-bench tg, 1 token à la fois, sans CUDA graphs)"; out ""
out "La fusion des portes est active si la ligne \`MUL_MAT f32 2048x32x1 cols=1x1 +fused\` apparaît (30 appels par token)."
out ""
profil "référence" "$REF"
profil "05" ""

# --- 5. llama-bench
out "## 5. llama-bench tg128 (-fa on, 3 répétitions)"; out ""
out "| variante | tg128 t/s |"; out "|---|---|"
lb() { # lb <nom> <env>
  reste || return
  local r; r="$(run_env "$2" timeout 300 "$BIN/llama-bench" -m "$Q40" -ngl 99 -fa on -p 0 -n 128 -r 3 -o md 2> "$LOGS/lb.err" | grep -E 'tg128' | awk -F'|' '{print $(NF-1)}' | sed 's/^ *//; s/ *$//')"
  out "| $1 | ${r:-échec} |"
}
lb "référence" "$REF"
lb "norme q/k seule" "GGML_CUDA_PASCAL_FUSE_GDN_GATES=0"
lb "portes seules" "LLAMA_PASCAL_GDN_QK_NORM=0"
lb "05 (les deux)" ""
lb "référence (2e passage)" "$REF"
lb "05 (2e passage)" ""
out ""

# --- 6. génération réelle
out "## 6. Génération réelle (llama-server, q40, -fa on, -c $CTX, 600 tokens, temp 0)"; out ""
out "| variante | prompt | tokens prompt | tokens générés | tg t/s | brouillons acceptés/proposés |"
out "|---|---|---|---|---|---|"
if [ -f "$MTP_MODEL" ] && [ -s "$TOKENS" ]; then
  L="GGML_CUDA_PASCAL_HEAD_TOKENS=$TOKENS"
  MTP2=(-md "$MTP_MODEL" --spec-type draft-mtp --spec-draft-n-max 2)
  essai "référence (04 + liste), MTP2" "$REF $L" "${MTP2[@]}"
  essai "05 + liste, MTP2" "$L" "${MTP2[@]}"
  essai "référence (2e passage), MTP2" "$REF $L" "${MTP2[@]}"
  essai "05 + liste (2e passage), MTP2" "$L" "${MTP2[@]}"
  essai "05 norme q/k seule + liste, MTP2" "GGML_CUDA_PASCAL_FUSE_GDN_GATES=0 $L" "${MTP2[@]}"
else
  out "| MTP | — | modèle MTP ou liste de tokens absents ($MTP_MODEL, $TOKENS) | | | |"
fi
essai "référence, sans MTP" "$REF"
essai "05, sans MTP" ""
out ""
out "_Terminé : $(date -u '+%Y-%m-%d %H:%M UTC'), durée $((SECONDS - T_DEBUT)) s. Journaux dans $LOGS._"
