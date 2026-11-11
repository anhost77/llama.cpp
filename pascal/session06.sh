#!/usr/bin/env bash
# session06.sh — séance courte sur une P40 louée :
#   0. micro-banc « bulles » (étape 0 de MEGAKERNEL.md) : noyaux en série contre noyau persistant, sans llama.cpp ;
#   1. un seul build (BASE + 01 + 02 + 04 + 05 + 06), justesse, profil, llama-bench ;
#   2. génération réelle sur les requêtes de vérification de l'atelier, avec et sans les deux nouveautés de 06 ;
#   3. dump du graphe en mode serveur (vérification MTP à 3 colonnes), qui manquait.
# Cible : Ubuntu 22.04 + CUDA 12.4 (image nvidia/cuda:12.4.1-devel-ubuntu22.04), exécuté en root.
#
# Usage : BUDGET=2400 bash session06.sh [liste de tokens]      (défaut : $MODELS/tokens2.txt)
# Attendu : $MODELS/q40.gguf, $MODELS/mtp-q40.gguf, $MODELS/verif-reelles.jsonl, la liste de tokens, et à côté de ce
#           script : BASE, 01-prs.patch, 02-fork.patch, 04-mtp-liste.patch, 05-fusion.patch, 06-etat-direct.patch, bulles.cu
#
# Référence = même build, nouveautés de 06 coupées (mêmes chemins de code que 05) :
#   GGML_CUDA_PASCAL_GDN_DIRECT_STATE=0 GGML_CUDA_PASCAL_FUSE_SHEXP_GATE=0
set -u

MODELS="${MODELS:-/root/models}"
Q40="${Q40:-$MODELS/q40.gguf}"
MTP_MODEL="${MTP_MODEL:-$MODELS/mtp-q40.gguf}"
REQS="${REQS:-$MODELS/verif-reelles.jsonl}"
TOKENS="${1:-${TOKENS:-$MODELS/tokens2.txt}}"
OUT="${OUT:-/root/web/fork6.txt}"
WEB="$(dirname "$OUT")"
WORK="${WORK:-/root/fork6}"
PATCHES="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="${REPO_URL:-https://github.com/ggml-org/llama.cpp}"
PORT="${PORT:-8089}"
CTX="${CTX:-40000}"
BUDGET="${BUDGET:-2400}"
[ "$BUDGET" -gt 2400 ] && BUDGET=2400
LOGS="$WORK/logs"
SRC="$WORK/p06"
BIN="$SRC/build/bin"
T_DEBUT=$SECONDS
REF="GGML_CUDA_PASCAL_GDN_DIRECT_STATE=0 GGML_CUDA_PASCAL_FUSE_SHEXP_GATE=0"
ETAT="GGML_CUDA_PASCAL_FUSE_SHEXP_GATE=0"          # lecture directe de l'état seule
PORTE="GGML_CUDA_PASCAL_GDN_DIRECT_STATE=0"        # fusion de la fin du bloc MoE seule

export PATH="/usr/local/cuda/bin:/usr/local/cuda-12.4/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "$LOGS" "$WEB"
out() { printf '%s\n' "$*" >> "$OUT"; }
log() { printf '[%s +%ss] %s\n' "$(date +%H:%M:%S)" "$((SECONDS - T_DEBUT))" "$*" >&2; }
reste() { [ $((SECONDS - T_DEBUT)) -lt "$BUDGET" ]; }
run_env() { local e="$1"; shift; if [ -n "$e" ]; then env $e "$@"; else "$@"; fi; }

: > "$OUT"
out "# Fork Pascal — séance 06 (état du delta-net lu dans le cache, fin du bloc MoE fusionnée, micro-banc « bulles »)"
out ""
out "- Date : $(date -u '+%Y-%m-%d %H:%M UTC')"
out "- GPU : $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1)"
out "- Base : $(tr -d ' \r\n' < "$PATCHES/BASE") + 01 + 02 + 04 + 05 + 06 ; liste de tokens : $TOKENS"
out "- Référence = même build, nouveautés coupées : \`$REF\`"
out ""

# ----------------------------------------------------------------------------- 0. micro-banc « bulles »
out "## 0. Micro-banc « bulles » : noyaux en série (A) contre noyau persistant (B)"; out ""
if [ -f "$PATCHES/bulles.cu" ]; then
  if nvcc -O3 -arch=sm_61 -o "$WORK/bulles" "$PATCHES/bulles.cu" > "$LOGS/bulles-build.log" 2>&1; then
    for args in "16 8" "32 8" "16 2" "8 8"; do
      out "### blocs par SM et lignes par prise : $args"; out ""
      # shellcheck disable=SC2086
      timeout 240 "$WORK/bulles" $args 15 >> "$OUT" 2> "$LOGS/bulles.err" || out "> bulles $args : échec (voir $LOGS/bulles.err : $(tail -1 "$LOGS/bulles.err" | cut -c1-150))"
      out ""
    done
  else
    out "> **échec de compilation de bulles.cu** :"; out '```'; tail -20 "$LOGS/bulles-build.log" >> "$OUT"; out '```'; out ""
  fi
else
  out "bulles.cu absent."; out ""
fi

# ----------------------------------------------------------------------------- 1. compilation
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
  for pch in 01-prs.patch 02-fork.patch 04-mtp-liste.patch 05-fusion.patch 06-etat-direct.patch; do
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

essai() { # essai <nom> <env> <args serveur...> : les requêtes réelles l'une après l'autre, puis la somme
  local nom="$1" e="$2"; shift 2
  reste || { out "| $nom | budget épuisé | | | | | |"; return; }
  local cle; cle="$(printf '%s' "$nom" | tr -c 'A-Za-z0-9' '_')"
  local d="$LOGS/verif-$cle"; mkdir -p "$d"
  log "essai $nom"
  if ! demarrer_serveur "$e" "$LOGS/srv-$cle.log" "$@"; then
    out "| $nom | serveur non démarré : $(grep -i -E 'error|failed|invalid' "$LOGS/srv-$cle.log" | tail -1 | cut -c1-110) | | | | | |"
    arreter_serveur; return
  fi
  local i=0 ligne
  while IFS= read -r ligne; do
    [ -n "$ligne" ] || continue
    i=$((i + 1))
    printf '%s' "$ligne" | jq -c '. + {cache_prompt: false}' > "$d/req-$i.json" 2>/dev/null
    curl -sS --max-time 300 -H 'Content-Type: application/json' --data-binary "@$d/req-$i.json" \
         "http://127.0.0.1:$PORT/v1/chat/completions" > "$d/rep-$i.json" 2>/dev/null
    jq -r '.choices[0].message.content // ""' "$d/rep-$i.json" > "$d/texte-$i.txt" 2>/dev/null
  done < "$REQS"
  arreter_serveur
  [ -d "$LOGS/verif-ref" ] || cp -r "$d" "$LOGS/verif-ref"
  python3 - "$d" "$LOGS/verif-ref" "$i" "$nom" >> "$OUT" 2>> "$LOGS/essai.err" <<'PY'
import json, sys
d, ref, n, nom = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
pn = pms = dn = da = ok = same = 0
qn = qms = 0
detail = []
for i in range(1, n + 1):
    try:
        t = json.load(open("%s/rep-%d.json" % (d, i)))["timings"]
    except Exception:
        detail.append("err"); continue
    ok += 1
    pn += t.get("predicted_n", 0); pms += t.get("predicted_ms", 0.0)
    qn += t.get("prompt_n", 0); qms += t.get("prompt_ms", 0.0)
    dn += t.get("draft_n", 0) or 0; da += t.get("draft_n_accepted", 0) or 0
    a = open("%s/texte-%d.txt" % (d, i), encoding="utf-8", errors="ignore").read()
    b = open("%s/texte-%d.txt" % (ref, i), encoding="utf-8", errors="ignore").read()
    same += a == b
    detail.append("%.0f" % t.get("predicted_per_second", 0))
tps = 1000.0 * pn / pms if pms > 0 else 0.0
pps = 1000.0 * qn / qms if qms > 0 else 0.0
print("| %s | %d/%d | %d | **%.1f** | %d/%d | %d/%d | %.0f t/s, %.1f s | %s |" % (nom, ok, n, pn, tps, da, dn, same, ok, pps, (pms + qms) / 1000.0, " ".join(detail)))
PY
}

profil() { # profil <nom> <env> : llama-bench sous GGML_CUDA_PROFILE, 22 premières lignes
  local f="$LOGS/profil-$1"
  run_env "$2 GGML_CUDA_PROFILE=1 GGML_CUDA_DISABLE_GRAPHS=1" timeout 200 "$BIN/llama-bench" -m "$Q40" -ngl 99 -fa on -p 0 -n 96 -r 1 \
      > "$f.out" 2> "$f.err"
  out "### $1"; out ""; out '```'
  grep -F '[ggml-cuda profile]' "$f.err" | head -24 >> "$OUT"
  out "GET_ROWS : $(grep -F '[ggml-cuda profile]' "$f.err" | grep -F ' GET_ROWS' | awk '{printf "%s appels, %s us/appel", $5, $6}')"
  out "nœuds par token (somme des appels / 97 graphes) : $(grep -F '[ggml-cuda profile]' "$f.err" | awk 'NR > 2 { n += $5 } END { printf "%.0f", n / 97 }')"
  out '```'; out ""
}

# ----------------------------------------------------------------------------- déroulé
compiler || { out "_Arrêt : pas de build._"; exit 1; }
[ -f "$Q40" ] || { out "modèle $Q40 absent."; exit 1; }

# --- 2. justesse
out "## 2. Justesse"; out ""
if [ -x "$BIN/test-backend-ops" ]; then
  timeout 900 "$BIN/test-backend-ops" test -b CUDA0 -o GATED_DELTA_NET,GET_ROWS > "$LOGS/tbo.log" 2>&1
  rc_tbo=$?
  out "- test-backend-ops GATED_DELTA_NET, GET_ROWS : $(grep -E '[0-9]+/[0-9]+ tests passed' "$LOGS/tbo.log" | tail -1 | sed 's/^ *//') (code $rc_tbo)"
fi
det() { # det <nom> <env> : sortie déterministe, 1 token à la fois
  run_env "$2" timeout 300 "$BIN/llama-completion" -m "$Q40" -ngl 99 -fa on -c 4096 -no-cnv -n 200 --temp 0 -s 42 \
      --no-display-prompt --simple-io -p "Voici le début d'un roman policier se déroulant à Lyon en hiver. Chapitre premier. Le commissaire" \
      < /dev/null > "$LOGS/det-$1.txt" 2> "$LOGS/det-$1.err"
}
if [ -x "$BIN/llama-completion" ]; then
  det ref "$REF"; det etat "$ETAT"; det porte "$PORTE"; det tout ""
  for v in etat porte tout; do
    if cmp -s "$LOGS/det-ref.txt" "$LOGS/det-$v.txt"; then r="identique à la référence ($(wc -c < "$LOGS/det-ref.txt") octets)"
    else r="diffère de la référence à l'octet $(cmp -l "$LOGS/det-ref.txt" "$LOGS/det-$v.txt" 2>/dev/null | head -1 | awk '{print $1}') sur $(wc -c < "$LOGS/det-ref.txt")"; fi
    out "- sortie déterministe (200 tokens, temp 0), variante $v : $r"
  done
  out "  (attendu : « etat » identique au bit près, la lecture directe ne change aucun calcul ; « porte » peut diverger, l'ordre des sommes change)"
fi
PPL_TXT="$WORK/ppl.txt"
if [ ! -s "$PPL_TXT" ]; then
  if curl -fsSL --max-time 120 -o "$WORK/wikitext.zip" "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip" 2>/dev/null \
     && unzip -o -q "$WORK/wikitext.zip" -d "$WORK" 2>/dev/null && [ -s "$WORK/wikitext-2-raw/wiki.test.raw" ]; then
    head -c 200000 "$WORK/wikitext-2-raw/wiki.test.raw" > "$PPL_TXT"
  fi
fi
ppl() { # ppl <env> <taille de bloc> <blocs>
  run_env "$1" timeout 600 "$BIN/llama-perplexity" -m "$Q40" -ngl 99 -fa off -c "$2" -b 2048 --chunks "$3" -f "$PPL_TXT" 2>&1 \
    | grep -E 'Final estimate' | tail -1 | sed -E 's/.*PPL = ([0-9.]+).*\+\/- ([0-9.]+).*/\1 ± \2/' | grep . || printf 'n/d'
}
if [ -s "$PPL_TXT" ] && [ -x "$BIN/llama-perplexity" ]; then
  out "- perplexité, 16 blocs de 512 tokens (master : 7.1616 ± 0.28264) : référence $(ppl "$REF" 512 16) ; 06 $(ppl "" 512 16)"
  out "- perplexité, 120 blocs de 64 tokens (séance 05 : 12.7132 / 12.7137 ; exerce la fusion de fin de bloc MoE) : référence $(ppl "$REF" 64 120) ; 06 $(ppl "" 64 120)"
else
  out "- perplexité : wikitext-2 indisponible, contrôle sauté"
fi
out ""

# --- 3. profil
out "## 3. Profil par type de nœud (llama-bench tg, 1 token à la fois, sans CUDA graphs)"; out ""
out "Attendu avec 06 : GET_ROWS passe de 62 à 32 appels par token (les 30 copies d'état de 2 Mo disparaissent),"
out "et une ligne \`MUL_MAT f32 2048x1x1 cols=1x1 +fused\` apparaît (40 appels par token)."
out ""
profil "référence" "$REF"
profil "06" ""

# --- 4. llama-bench
out "## 4. llama-bench tg128 (-fa on, 3 répétitions)"; out ""
out "| variante | tg128 t/s |"; out "|---|---|"
lb() { # lb <nom> <env>
  reste || return
  local r; r="$(run_env "$2" timeout 300 "$BIN/llama-bench" -m "$Q40" -ngl 99 -fa on -p 0 -n 128 -r 3 -o md 2> "$LOGS/lb.err" | grep -E 'tg128' | awk -F'|' '{print $(NF-1)}' | sed 's/^ *//; s/ *$//')"
  out "| $1 | ${r:-échec} |"
}
lb "référence" "$REF"
lb "état direct seul" "$ETAT"
lb "fin de bloc MoE seule" "$PORTE"
lb "06 (les deux)" ""
lb "référence (2e passage)" "$REF"
lb "06 (2e passage)" ""
out ""

# --- 5. génération réelle : requêtes de vérification de l'atelier
out "## 5. Requêtes réelles de vérification (llama-server, /v1/chat/completions, temp 0, liste de tokens)"; out ""
if [ -s "$REQS" ] && [ -f "$MTP_MODEL" ] && [ -s "$TOKENS" ]; then
  out "t/s = somme des tokens générés / somme des temps de génération ; « textes identiques » : par rapport à la première ligne."
  out ""
  out "| variante | réponses | tokens générés | tg t/s (somme) | brouillons acceptés/proposés | textes identiques | prompt : t/s, temps total des 8 requêtes | t/s par requête |"
  out "|---|---|---|---|---|---|---|---|"
  L="GGML_CUDA_PASCAL_HEAD_TOKENS=$TOKENS"
  M=(-md "$MTP_MODEL" --spec-draft-n-max 2 --spec-type draft-mtp)
  essai "référence (05), MTP2" "$REF $L" "${M[@]}"
  essai "06, MTP2" "$L" "${M[@]}"
  essai "06 état direct seul, MTP2" "$ETAT $L" "${M[@]}"
  essai "référence (2e passage), MTP2" "$REF $L" "${M[@]}"
  essai "06 (2e passage), MTP2" "$L" "${M[@]}"
  essai "référence, sans spéculatif" "$REF $L"
  essai "06, sans spéculatif" "$L"
else
  out "requêtes ($REQS), modèle MTP ou liste de tokens absents : étape sautée."
fi
out ""

# --- 6. dumps de graphe : génération (llama-bench) et serveur avec MTP (vérification à 3 colonnes, instantanés)
out "## 6. Listes de nœuds"; out ""
run_env "GGML_CUDA_PASCAL_DUMP_GRAPH=6 GGML_CUDA_PASCAL_DUMP_FILE=$WEB/graphe06-bench.txt" \
    timeout 200 "$BIN/llama-bench" -m "$Q40" -ngl 99 -fa on -p 0 -n 12 -r 1 > "$LOGS/dump-bench.out" 2> "$LOGS/dump-bench.err"
out "- llama-bench, 1 token : $( [ -s "$WEB/graphe06-bench.txt" ] && echo "$(wc -l < "$WEB/graphe06-bench.txt") nœuds, $WEB/graphe06-bench.txt" || echo "fichier non produit" )"
if reste && [ -s "$REQS" ] && [ -f "$MTP_MODEL" ]; then
  # le 40e graphe d'au moins 200 nœuds évalué par le serveur : après le prompt, donc un pas de vérification MTP
  if demarrer_serveur "GGML_CUDA_PASCAL_DUMP_GRAPH=40 GGML_CUDA_PASCAL_DUMP_FILE=$WEB/graphe06-serveur-mtp.txt" "$LOGS/srv-dump.log" \
        -md "$MTP_MODEL" --spec-draft-n-max 2 --spec-type draft-mtp; then
    head -1 "$REQS" | jq -c '. + {cache_prompt: false, max_tokens: 120}' > "$WORK/req-dump.json"
    curl -sS --max-time 200 -H 'Content-Type: application/json' --data-binary "@$WORK/req-dump.json" \
         "http://127.0.0.1:$PORT/v1/chat/completions" > /dev/null 2>&1
  fi
  arreter_serveur
  out "- serveur + MTP n-max 2, 40e graphe : $( [ -s "$WEB/graphe06-serveur-mtp.txt" ] && echo "$(wc -l < "$WEB/graphe06-serveur-mtp.txt") nœuds, $WEB/graphe06-serveur-mtp.txt ; première ligne MUL_MAT : $(grep -m1 ' MUL_MAT ' "$WEB/graphe06-serveur-mtp.txt" | cut -c1-90)" || echo "fichier non produit" )"
fi
out ""
out "_Terminé : $(date -u '+%Y-%m-%d %H:%M UTC'), durée $((SECONDS - T_DEBUT)) s. Journaux dans $LOGS._"
