# Digest-on-exit for the dfs6311 SLURM jobs: compose the job's own error
# evidence (rc, error-context lines, log tail, report tail) into one
# plain-text digest and mail it, so a failure can be read -- and acted on --
# from the email alone, without cluster shell access. SLURM's --mail-type
# notifications carry job STATE only; this carries the content.
#
# Usage (from an sbatch script, after `set -uo pipefail` and after the log
# path variables are defined):
#
#   source "$REPO/hpcjobs/job_digest.sh"
#   job_digest_arm TAG RECIPIENT LOGFILE [REPORT_GLOB]
#
# The EXIT trap then fires on every path out of the script -- success, stage
# failure, scancel, or the wall limit (SLURM sends TERM before KILL, and the
# TERM trap sends the digest inside that grace window). The digest is also
# written to ${LOGFILE}.digest.txt so the content survives even when no
# mailer is available on the node. Every step is soft-failed: the digest
# machinery must never change the job's exit status.

job_digest_send() {
  # $1 tag, $2 recipient, $3 rc, $4 logfile, $5 report glob (may be empty)
  local tag="${1:-job}" to="${2:-}" rc="${3:-unknown}" log="${4:-}" rep="${5:-}"
  local digest="${log:-/tmp/job}.digest.txt"
  {
    echo "=== ${tag} job ${SLURM_JOB_ID:-manual} on $(hostname 2>/dev/null || echo unknown) ==="
    echo "rc=${rc}  ended $(date '+%Y-%m-%d %H:%M:%S %Z' 2>/dev/null || true)"
    echo
    if [ -n "$log" ] && [ -f "$log" ]; then
      echo "--- error context (pattern matches, 3 lines after, first 80) ---"
      grep -nE -A3 'Traceback|[A-Za-z]+Error|FATAL|FAILURE|non-finite|Killed|CANCELLED|oom-kill|OOM' \
        "$log" 2>/dev/null | head -80 || echo "(no error-pattern matches)"
      echo
      echo "--- last 100 log lines ---"
      tail -100 "$log" 2>/dev/null || true
    else
      echo "(log file not found: ${log:-unset})"
    fi
    if [ -n "$rep" ]; then
      # shellcheck disable=SC2086 -- unquoted on purpose: the report path is
      # passed as a glob and expanded here.
      for f in $rep; do
        [ -f "$f" ] || continue
        echo
        echo "--- report tail: $f ---"
        tail -40 "$f" 2>/dev/null || true
      done
    fi
  } > "$digest" 2>/dev/null || true

  local subj="[${tag} job ${SLURM_JOB_ID:-manual}] rc=${rc}"
  if [ -n "$to" ] && [ -f "$digest" ]; then
    if command -v mail >/dev/null 2>&1; then
      mail -s "$subj" "$to" < "$digest" 2>/dev/null || true
    elif command -v mailx >/dev/null 2>&1; then
      mailx -s "$subj" "$to" < "$digest" 2>/dev/null || true
    elif command -v sendmail >/dev/null 2>&1; then
      { printf 'To: %s\nSubject: %s\n\n' "$to" "$subj"; cat "$digest"; } \
        | sendmail -t 2>/dev/null || true
    else
      echo "[digest] no mailer on this node; digest at $digest"
    fi
  fi
  return 0
}

job_digest_fire() {
  # Idempotent: the TERM path fires the digest and then exits, which would
  # otherwise fire the EXIT trap a second time and send a duplicate email.
  [ "${JOB_DIGEST_DONE:-0}" -eq 1 ] && return 0
  JOB_DIGEST_DONE=1
  job_digest_send "${JOB_DIGEST_TAG:-job}" "${JOB_DIGEST_TO:-}" "${1:-unknown}" \
    "${JOB_DIGEST_LOG:-}" "${JOB_DIGEST_REPORT:-}"
  return 0
}

job_digest_arm() {
  # $1 tag, $2 recipient, $3 logfile, $4 report glob (optional)
  JOB_DIGEST_TAG="${1:-job}"
  JOB_DIGEST_TO="${2:-}"
  JOB_DIGEST_LOG="${3:-}"
  JOB_DIGEST_REPORT="${4:-}"
  JOB_DIGEST_DONE=0
  trap 'job_digest_fire $?' EXIT
  trap 'echo "[digest] TERM received (scancel or wall limit)"; job_digest_fire 143; exit 143' TERM
  return 0
}
