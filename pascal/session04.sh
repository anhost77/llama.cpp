#!/usr/bin/env bash
# session04.sh — séance courte (≤ 45 min) sur une P40 louée : un seul build (BASE + 01 + 02 + 04), profil par
# type de nœud, puis génération réelle llama-server avec MTP n-max 2, avec et sans les bascules de 04.
# Cible : Ubuntu 22.04 + CUDA 12.4 (image nvidia/cuda:12.4.1-devel-ubuntu22.04), exécuté en root.
#
# Attendu sur la machine :
#   $MODELS/q40.gguf       Qwen_Qwen3.6-35B-A3B-Q4_0.gguf (bartowski)
#   $MODELS/mtp-q40.gguf   mtp-Qwen_Qwen3.6-35B-A3B-Q4_0.gguf
#   $MODELS/prompt.txt     le vrai prompt (chapitres)
#   à côté de ce script : BASE, 01-prs.patch, 02-fork.patch, 04-mtp-liste.patch
# Optionnel : $MODELS/corpus.txt (texte français, autre que le prompt) pour la liste de tokens.
#
# Le build 04 sans variable d'environnement = p02 avec 2 lignes par bloc. La référence « p02 » est obtenue avec
# GGML_CUDA_PASCAL_MMVQ_ROWS=-1 (choix par K, le défaut de p02) et sans liste de tokens : mêmes chemins de code que p02.
set -u

MODELS="${MODELS:-/root/models}"
Q40="${Q40:-$MODELS/q40.gguf}"
MTP_MODEL="${MTP_MODEL:-$MODELS/mtp-q40.gguf}"
PROMPT_FILE="${PROMPT_FILE:-$MODELS/prompt.txt}"
CORPUS_FILE="${CORPUS_FILE:-$MODELS/corpus.txt}"
OUT="${OUT:-/root/web/fork4.txt}"
WORK="${WORK:-/root/fork4}"
PATCHES="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="${REPO_URL:-https://github.com/ggml-org/llama.cpp}"
PORT="${PORT:-8089}"
CTX="${CTX:-40000}"
BUDGET="${BUDGET:-2700}"          # secondes : au-delà, les essais restants sont sautés
LOGS="$WORK/logs"
SRC="$WORK/p04"
BIN="$SRC/build/bin"
TOKENS="$WORK/tokens.txt"
T_DEBUT=$SECONDS

export PATH="/usr/local/cuda/bin:/usr/local/cuda-12.4/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "$LOGS" "$(dirname "$OUT")"
out() { printf '%s\n' "$*" >> "$OUT"; }
log() { printf '[%s +%ss] %s\n' "$(date +%H:%M:%S)" "$((SECONDS - T_DEBUT))" "$*" >&2; }
reste() { [ $((SECONDS - T_DEBUT)) -lt "$BUDGET" ]; }
run_env() { local e="$1"; shift; if [ -n "$e" ]; then env $e "$@"; else "$@"; fi; }

: > "$OUT"
out "# Fork Pascal — séance 04 (liste de tokens pour la tête MTP, profil)"
out ""
out "- Date : $(date -u '+%Y-%m-%d %H:%M UTC')"
out "- GPU : $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1)"
out "- Base : $(tr -d ' \r\n' < "$PATCHES/BASE") + 01 + 02 + 04"
out ""

# ----------------------------------------------------------------------------- compilation
compiler() {
  out "## 1. Compilation"; out ""
  export DEBIAN_FRONTEND=noninteractive
  local manquants="" p
  for p in git cmake g++ make curl jq python3; do command -v "$p" >/dev/null 2>&1 || manquants="$manquants $p"; done
  if [ -n "$manquants" ] || [ ! -e /usr/include/x86_64-linux-gnu/curl/curl.h ]; then
    log "installation des paquets (manquants :$manquants)"
    apt-get update -y > "$LOGS/apt.log" 2>&1
    apt-get install -y git cmake build-essential curl jq python3 libcurl4-openssl-dev libssl-dev >> "$LOGS/apt.log" 2>&1
  fi
  local base; base="$(tr -d ' \r\n' < "$PATCHES/BASE")"
  if [ ! -x "$BIN/llama-server" ]; then
    if [ ! -d "$SRC/.git" ]; then
      log "clonage"
      git clone "$REPO_URL" "$SRC" > "$LOGS/clone.log" 2>&1 || { out "> ERREUR : clonage impossible"; return 1; }
    fi
    git -C "$SRC" cat-file -e "$base^{commit}" 2>/dev/null || git -C "$SRC" fetch origin "$base" >> "$LOGS/clone.log" 2>&1
    git -C "$SRC" checkout -q -f "$base" >> "$LOGS/clone.log" 2>&1 || { out "> ERREUR : commit $base introuvable"; return 1; }
    git -C "$SRC" reset -q --hard "$base"
    local pch
    for pch in 01-prs.patch 02-fork.patch 04-mtp-liste.patch; do
      if ! git -C "$SRC" apply --whitespace=nowarn "$PATCHES/$pch" 2> "$LOGS/apply-$pch.log"; then
        out "> ERREUR : \`git apply $pch\` a échoué :"; out '```'; tail -20 "$LOGS/apply-$pch.log" >> "$OUT"; out '```'
        return 1
      fi
    done
    log "compilation"
    local t0=$SECONDS
    cmake -S "$SRC" -B "$SRC/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_BUILD_TYPE=Release \
          > "$LOGS/cmake.log" 2>&1
    cmake --build "$SRC/build" -j "$(nproc)" --target llama-bench llama-server > "$LOGS/build.log" 2>&1
    if [ -x "$BIN/llama-server" ] && [ -x "$BIN/llama-bench" ]; then
      out "- compilé en $((SECONDS - t0)) s"
    else
      out "- **ÉCHEC de compilation** :"
      out '```'; grep -n -i -B2 -A6 'error' "$LOGS/build.log" | head -80 >> "$OUT"; tail -15 "$LOGS/build.log" >> "$OUT"; out '```'
      return 1
    fi
  else
    out "- déjà compilé"
  fi
  out ""
}

# ----------------------------------------------------------------------------- serveur
SRV_PID=""
arreter_serveur() {
  if [ -n "$SRV_PID" ]; then
    kill "$SRV_PID" 2>/dev/null
    local i; for i in $(seq 1 20); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
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

MTP2=(-md "$MTP_MODEL" --spec-type draft-mtp --spec-draft-n-max 2)

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
  grep -h "the MTP head computes" "$journal" 2>/dev/null | tail -1 | sed 's/^/> /' >> "$OUT"
}

# ----------------------------------------------------------------------------- liste de tokens
# Fréquences des tokens d'un corpus français qui n'est PAS le prompt de test (sinon la mesure serait biaisée).
construire_liste() {
  out "## 3. Liste de tokens pour la tête MTP"; out ""
  mkdir -p "$WORK/corpus"
  if [ -s "$CORPUS_FILE" ]; then
    cp "$CORPUS_FILE" "$WORK/corpus/corpus.txt"
  else
    # romans français du domaine public (Projet Gutenberg) ; les téléchargements ratés sont ignorés
    local id
    for id in 17489 13951 14155 5097 798 4650 2650; do
      curl -fsSL --max-time 40 "https://www.gutenberg.org/cache/epub/$id/pg$id.txt" -o "$WORK/corpus/pg$id.txt" 2>/dev/null
      [ "$(stat -c %s "$WORK/corpus/pg$id.txt" 2>/dev/null || echo 0)" -gt 100000 ] || rm -f "$WORK/corpus/pg$id.txt"
    done
  fi
  local biais=0
  if ! ls "$WORK/corpus"/*.txt >/dev/null 2>&1; then
    cp "$PROMPT_FILE" "$WORK/corpus/prompt.txt"; biais=1
    out "> **ATTENTION** : aucun corpus externe disponible, la liste est construite sur le prompt de test lui-même :"
    out "> le taux d'acceptation mesuré ci-dessous est donc optimiste."
  fi
  if ! demarrer_serveur "" "$LOGS/srv-tokenize.log"; then
    out "> ERREUR : serveur non démarré pour la tokenisation"; arreter_serveur; return 1
  fi
  python3 - "$PORT" "$TOKENS" "$PROMPT_FILE" "$WORK/corpus"/*.txt >> "$OUT" 2> "$LOGS/tokens.err" <<'PY'
import collections, json, sys, urllib.request
port, sortie, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
fichiers = sys.argv[4:]

def tokeniser(texte, special=False):
    corps = json.dumps({"content": texte, "add_special": False, "parse_special": special}).encode()
    req = urllib.request.Request("http://127.0.0.1:%s/tokenize" % port, data=corps, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))["tokens"]

compte = collections.Counter()
total_car = 0
for f in fichiers:
    texte = open(f, encoding="utf-8", errors="ignore").read()
    if "gutenberg" in texte[:3000].lower():          # retirer l'en-tête et la licence Gutenberg (anglais)
        d = texte.find("*** START"); fin = texte.find("*** END")
        if d >= 0: texte = texte[texte.find("\n", d) + 1:]
        if fin >= 0: texte = texte[:texte.find("*** END")]
    total_car += len(texte)
    for i in range(0, len(texte), 200000):
        compte.update(tokeniser(texte[i:i + 200000]))
n_corpus = sum(compte.values())
ordre = [t for t, _ in compte.most_common()]
# toujours présents : fin de tour / fin de texte, sauts de ligne, puis les premiers identifiants (octets, ponctuation)
tete = []
for s in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
    try: tete += tokeniser(s, True)
    except Exception: pass
tete += tokeniser("\n") + tokeniser("\n\n")
vus, liste = set(), []
for t in tete + ordre + list(range(0, 1024)):
    if t not in vus:
        vus.add(t); liste.append(t)
open(sortie, "w").write("\n".join(map(str, liste)) + "\n")

p = tokeniser(open(prompt, encoding="utf-8", errors="ignore").read())
print("- corpus : %d fichier(s), %d caractères, %d tokens, %d tokens distincts ; liste écrite : %d identifiants" % (len(fichiers), total_car, n_corpus, len(compte), len(liste)))
print("- couverture des tokens du prompt de test (%d tokens) par les N premiers de la liste :" % len(p))
for n in (4096, 8192, 16384, 32768, 65536):
    s = set(liste[:n])
    print("  - N = %d : %.2f %%" % (n, 100.0 * sum(1 for t in p if t in s) / max(1, len(p))))
PY
  [ -s "$TOKENS" ] || { out "> ERREUR : liste de tokens non produite ($(tail -1 "$LOGS/tokens.err" | cut -c1-160))"; arreter_serveur; return 1; }
  arreter_serveur
  out ""
  return 0
}

# ----------------------------------------------------------------------------- déroulé
compiler || { out "_Arrêt : pas de build._"; exit 1; }
[ -f "$Q40" ] || { out "modèle $Q40 absent."; exit 1; }
[ -f "$MTP_MODEL" ] || { out "modèle MTP $MTP_MODEL absent."; exit 1; }
[ -s "$PROMPT_FILE" ] || { out "prompt $PROMPT_FILE absent."; exit 1; }
{ head -c 8000   "$PROMPT_FILE"; printf '\n\nChapitre suivant.\n\n'; } > "$WORK/prompt-court.txt"
{ head -c 120000 "$PROMPT_FILE"; printf '\n\nChapitre suivant.\n\n'; } > "$WORK/prompt-long.txt"

out "## 2. Profil par type de nœud (llama-bench, génération 1 token à la fois, sans CUDA graphs)"; out ""
out "Temps mur entre deux synchronisations du flux : les très petits noyaux sont un peu surestimés. Les 30 premières lignes :"
out '```'
GGML_CUDA_PROFILE=1 GGML_CUDA_DISABLE_GRAPHS=1 timeout 300 "$BIN/llama-bench" -m "$Q40" -ngl 99 -fa on -p 0 -n 96 -r 1 \
    > "$LOGS/profil-bench.out" 2> "$LOGS/profil-bench.err"
grep -F '[ggml-cuda profile]' "$LOGS/profil-bench.err" | head -32 >> "$OUT"
out '```'
out ""

construire_liste; LISTE_OK=$?

out "## 4. Génération réelle (llama-server, q40, -fa on, -c $CTX, 600 tokens, temp 0, MTP n-max 2 sauf mention)"; out ""
out "| variante | prompt | tokens prompt | tokens générés | tg t/s | brouillons acceptés/proposés |"
out "|---|---|---|---|---|---|"
REF="GGML_CUDA_PASCAL_MMVQ_ROWS=-1"
essai "référence p02 (rows par K), MTP2" "$REF" "${MTP2[@]}"
essai "04 défaut (2 lignes), MTP2" "" "${MTP2[@]}"
if [ "$LISTE_OK" -eq 0 ]; then
  for n in 16384 32768 8192 65536; do
    essai "04 liste N=$n, MTP2" "GGML_CUDA_PASCAL_HEAD_TOKENS=$TOKENS GGML_CUDA_PASCAL_HEAD_N=$n" "${MTP2[@]}"
  done
fi
essai "référence p02 (2e passage), MTP2" "$REF" "${MTP2[@]}"
if [ "$LISTE_OK" -eq 0 ]; then
  essai "04 liste N=16384 (2e passage), MTP2" "GGML_CUDA_PASCAL_HEAD_TOKENS=$TOKENS GGML_CUDA_PASCAL_HEAD_N=16384" "${MTP2[@]}"
  essai "04 liste N=16384, MTP n-max 3" "GGML_CUDA_PASCAL_HEAD_TOKENS=$TOKENS GGML_CUDA_PASCAL_HEAD_N=16384" \
      -md "$MTP_MODEL" --spec-type draft-mtp --spec-draft-n-max 3
fi
essai "04 défaut, sans MTP" ""
out ""

if reste; then
  out "## 5. Profil d'un pas MTP n-max 2 (llama-server, prompt court, sans CUDA graphs)"; out ""
  out "Inclut le traitement du prompt (lignes à beaucoup de colonnes) ; lire surtout les lignes cols=1x1, 2x1, 3x1, 1x2, 1x3."
  if demarrer_serveur "GGML_CUDA_PROFILE=1 GGML_CUDA_DISABLE_GRAPHS=1" "$LOGS/srv-profil-mtp.log" "${MTP2[@]}"; then
    out "- tg sous profil (non comparable) : $(requete "$WORK/prompt-court.txt")"
  fi
  arreter_serveur
  out '```'
  grep -F '[ggml-cuda profile]' "$LOGS/srv-profil-mtp.log" | head -45 >> "$OUT"
  out '```'
  out ""
fi

out "_Terminé : $(date -u '+%Y-%m-%d %H:%M UTC'), durée $((SECONDS - T_DEBUT)) s. Journaux dans $LOGS._"
