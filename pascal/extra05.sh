#!/usr/bin/env bash
# extra05.sh — suite de session05.sh sur la même machine : brouillon par recherche de n-grammes dans le prompt
# (--spec-type ngram-*,draft-mtp, déjà dans llama.cpp à BASE) sur les vraies requêtes de vérification de l'atelier.
# Usage : bash extra05.sh [liste de tokens] ; requêtes : $MODELS/verif-reelles.jsonl (une requête /v1/chat/completions par ligne)
set -u
MODELS=/root/models; Q40=$MODELS/q40.gguf; MTP_MODEL=$MODELS/mtp-q40.gguf
TOKENS="${1:-$MODELS/tokens2.txt}"
REQS="${REQS:-$MODELS/verif-reelles.jsonl}"
WORK=/root/fork5; LOGS=$WORK/logs; BIN=$WORK/p05/build/bin; OUT=/root/web/fork5b.txt; PORT=8089; CTX=40000
BUDGET="${BUDGET:-840}"; T_DEBUT=$SECONDS
export PATH="/usr/local/cuda/bin:$PATH"; export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
out() { printf '%s\n' "$*" >> "$OUT"; }
reste() { [ $((SECONDS - T_DEBUT)) -lt "$BUDGET" ]; }
run_env() { local e="$1"; shift; if [ -n "$e" ]; then env $e "$@"; else "$@"; fi; }
SRV_PID=""
arreter_serveur() {
  if [ -n "$SRV_PID" ]; then kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null; fi
  pkill -f "[l]lama-server.*--port $PORT" 2>/dev/null
  local i; for i in $(seq 1 30); do pgrep -f "[l]lama-server.*--port $PORT" >/dev/null || break; sleep 1; done
  pkill -9 -f "[l]lama-server.*--port $PORT" 2>/dev/null
  for i in $(seq 1 15); do curl -fsS --max-time 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || break; sleep 1; done
  SRV_PID=""
}
trap arreter_serveur EXIT
demarrer_serveur() {
  local e="$1" journal="$2"; shift 2
  curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "port occupé" > "$journal"; return 1; }
  run_env "$e" "$BIN/llama-server" -m "$Q40" -ngl 99 -c "$CTX" -np 1 -ub 2048 -b 2048 -fa on \
      --host 127.0.0.1 --port "$PORT" --no-webui "$@" > "$journal" 2>&1 &
  SRV_PID=$!
  local i
  for i in $(seq 1 150); do
    kill -0 "$SRV_PID" 2>/dev/null || return 1
    curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && return 0
    sleep 2
  done
  return 1
}
L="GGML_CUDA_PASCAL_HEAD_TOKENS=$TOKENS"

essai() { # essai <nom> <args serveur...> : les requêtes réelles l'une après l'autre, puis la somme
  local nom="$1"; shift
  reste || { out "| $nom | budget épuisé | | | | | |"; return; }
  local cle; cle="$(printf '%s' "$nom" | tr -c 'A-Za-z0-9' '_')"
  local d="$LOGS/verif-$cle"; mkdir -p "$d"
  if ! demarrer_serveur "$L" "$LOGS/srvx-$cle.log" "$@"; then
    out "| $nom | serveur non démarré : $(grep -i -E 'error|failed|invalid' "$LOGS/srvx-$cle.log" | tail -1 | cut -c1-110) | | | | | |"
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
  python3 - "$d" "$LOGS/verif-ref" "$i" "$nom" >> "$OUT" 2>> "$LOGS/extra05.err" <<'PY'
import json, sys
d, ref, n, nom = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
pn = pms = dn = da = ok = same = 0
detail = []
for i in range(1, n + 1):
    try:
        t = json.load(open("%s/rep-%d.json" % (d, i)))["timings"]
    except Exception:
        detail.append("err"); continue
    ok += 1
    pn += t.get("predicted_n", 0); pms += t.get("predicted_ms", 0.0)
    dn += t.get("draft_n", 0) or 0; da += t.get("draft_n_accepted", 0) or 0
    a = open("%s/texte-%d.txt" % (d, i), encoding="utf-8", errors="ignore").read()
    b = open("%s/texte-%d.txt" % (ref, i), encoding="utf-8", errors="ignore").read()
    same += a == b
    detail.append("%.0f" % t.get("predicted_per_second", 0))
tps = 1000.0 * pn / pms if pms > 0 else 0.0
print("| %s | %d/%d | %d | **%.1f** | %d/%d | %d/%d | %s |" % (nom, ok, n, pn, tps, da, dn, same, ok, " ".join(detail)))
PY
}

: > "$OUT"
out "# Séance 05 — suite : recherche de n-grammes dans le prompt + MTP, requêtes de vérification réelles"; out ""
[ -s "$REQS" ] || { out "fichier $REQS absent."; exit 1; }
out "Requêtes : $(grep -c . "$REQS") requêtes réelles (\`$REQS\`), /v1/chat/completions, temp 0, telles quelles (+ cache_prompt false)."
out "Build 05 par défaut + liste de tokens. t/s = somme des tokens générés / somme des temps de génération."
out "« textes identiques » : sorties identiques à celles de la première ligne (référence)."
out ""
out "| variante | réponses | tokens générés | tg t/s (somme) | brouillons acceptés/proposés | textes identiques | t/s par requête |"
out "|---|---|---|---|---|---|---|"
M=(-md "$MTP_MODEL" --spec-draft-n-max 2)
essai "MTP2 seul (référence)"                 "${M[@]}" --spec-type draft-mtp
essai "ngram-simple (n=12, m=48) + MTP2"      "${M[@]}" --spec-type ngram-simple,draft-mtp
essai "ngram-simple n=6 m=32 seul (sans MTP)" --spec-type ngram-simple --spec-ngram-simple-size-n 6 --spec-ngram-simple-size-m 32
essai "ngram-simple n=6 m=32 + MTP2"          "${M[@]}" --spec-type ngram-simple,draft-mtp --spec-ngram-simple-size-n 6 --spec-ngram-simple-size-m 32
essai "ngram-mod (défauts) + MTP2"            "${M[@]}" --spec-type ngram-mod,draft-mtp
essai "ngram-simple n=4 m=16 + MTP2"          "${M[@]}" --spec-type ngram-simple,draft-mtp --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 16
essai "ngram-map-k (défauts) + MTP2"          "${M[@]}" --spec-type ngram-map-k,draft-mtp
essai "MTP2 seul (2e passage)"                "${M[@]}" --spec-type draft-mtp
out ""; out "_Terminé : $(date -u '+%H:%M UTC'), $((SECONDS - T_DEBUT)) s._"
