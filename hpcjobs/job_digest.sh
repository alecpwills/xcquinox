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
# TERM trap sends the digest inside that grace window). The digest body is
# composed IN MEMORY and handed to the mailer directly, so a full disk or an
# unwritable path cannot suppress the email and a stale on-disk digest can
# never be re-sent under a fresh subject; the ${LOGFILE}.digest.txt copy is
# best-effort. Every mailer call is bounded by a timeout: a hung mailer must
# not hold a finished job (and its exclusive node) until the wall limit.
# Every step is soft-failed: the digest machinery must never change the
# job's exit status.
#
# KNOWN LIMIT: bash defers traps while a foreground child runs, so a child
# that itself traps or ignores SIGTERM would stall the digest past SLURM's
# kill grace and nothing would be sent. None of the drivers this instruments
# installs a SIGTERM handler; keep it that way, or move the digest into the
# driver.

job_digest_compose() {
  # $1 tag, $2 rc, $3 logfile, $4 report glob -- digest text on stdout.
  local tag="${1:-job}" rc="${2:-unknown}" log="${3:-}" rep="${4:-}"
  echo "=== ${tag} job ${SLURM_JOB_ID:-manual} on $(hostname 2>/dev/null || echo unknown) ==="
  echo "rc=${rc}  ended $(date '+%Y-%m-%d %H:%M:%S %Z' 2>/dev/null || true)"
  echo
  if [ -n "$log" ] && [ -f "$log" ]; then
    # No -A context: matched lines only, so a long failure cascade is not
    # truncated by its own context lines. [A-Za-z]*Error also matches a
    # line-initial "Error:"; the explicit alternatives cover the killers
    # that carry no "Error" token at all (SIGSEGV/SIGABRT text, the OOM
    # killer, slurmstepd, glibc/libstdc++ aborts).
    local errs=""
    errs="$(grep -nE 'Traceback|[A-Za-z]*Error|FATAL|FAILURE|non-finite|Segmentation fault|Aborted|core dumped|Fatal Python error|bad_alloc|Exceeded .* memory|slurmstepd: error|Killed|CANCELLED|oom-kill|OOM' \
              "$log" 2>/dev/null | head -120 || true)"
    echo "--- error-pattern lines (first 120) ---"
    if [ -n "$errs" ]; then
      printf '%s\n' "$errs"
    else
      echo "(no error-pattern matches)"
    fi
    echo
    echo "--- last 100 log lines ---"
    tail -100 "$log" 2>/dev/null || true
  else
    echo "(log file not found: ${log:-unset})"
  fi
  if [ -n "$rep" ]; then
    local f
    # shellcheck disable=SC2086 -- unquoted on purpose: the report path is
    # passed as a glob and expanded here.
    for f in $rep; do
      [ -f "$f" ] || continue
      echo
      echo "--- report tail: $f ---"
      tail -40 "$f" 2>/dev/null || true
    done
  fi
  return 0
}

job_digest_send() {
  # $1 tag, $2 recipient, $3 rc, $4 logfile, $5 report glob (may be empty)
  local tag="${1:-job}" to="${2:-}" rc="${3:-unknown}" log="${4:-}" rep="${5:-}"
  local digest="${log:-/tmp/job}.digest.txt" body=""
  body="$(job_digest_compose "$tag" "$rc" "$log" "$rep" 2>/dev/null || true)"
  [ -n "$body" ] || body="=== ${tag} job ${SLURM_JOB_ID:-manual} === rc=${rc} (digest compose produced no content)"
  # Best-effort on-disk copy; the email below does NOT depend on it.
  printf '%s\n' "$body" > "$digest" 2>/dev/null || true

  local subj="[${tag} job ${SLURM_JOB_ID:-manual}] rc=${rc}"
  if [ -n "$to" ]; then
    local tmo=""
    command -v timeout >/dev/null 2>&1 \
      && tmo="timeout ${JOB_DIGEST_MAIL_TIMEOUT:-60}"
    if command -v mail >/dev/null 2>&1; then
      # shellcheck disable=SC2086 -- $tmo is deliberately word-split
      printf '%s\n' "$body" | $tmo mail -s "$subj" "$to" 2>/dev/null || true
    elif command -v mailx >/dev/null 2>&1; then
      # shellcheck disable=SC2086
      printf '%s\n' "$body" | $tmo mailx -s "$subj" "$to" 2>/dev/null || true
    else
      # sendmail commonly lives in /usr/sbin, which batch PATHs may lack.
      local sm=""
      if command -v sendmail >/dev/null 2>&1; then sm="sendmail"
      elif [ -x /usr/sbin/sendmail ]; then sm="/usr/sbin/sendmail"; fi
      if [ -n "$sm" ]; then
        # shellcheck disable=SC2086
        { printf 'To: %s\nSubject: %s\n\n' "$to" "$subj"
          printf '%s\n' "$body"; } | $tmo "$sm" -t 2>/dev/null || true
      else
        echo "[digest] no mailer on this node; digest at $digest"
      fi
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
