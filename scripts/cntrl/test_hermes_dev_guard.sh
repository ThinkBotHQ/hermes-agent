#!/bin/zsh
# Proof harness for the hermes-dev launch guard (scripts/cntrl/hermes-dev).
#
# Sources the launcher with HERMES_DEV_SOURCE_ONLY=1 (nothing is synced or launched) and runs
# hermes_dev_guard against throwaway worktrees at three real commits, with a stubbed `gh`
# that serves canned check-run JSON:
#   c0deb07865  broke the Electron main type-check        -> refuse (build check)
#   c0deb0749c  type-checks, crashed on every launch      -> refuse (CI launch smoke failed)
#   c0deb0715b  the fix                                   -> pass
# plus the no-rebuild cache and the other refusals (smoke still running, gh offline, no smoke
# check yet, worktree not at T, skipped smoke with desktop changes).
#
# The worktrees borrow node_modules by symlink from the ship worktree (no installs) and are
# removed on exit. Exit status is non-zero if any case does not match its expectation.
set -u
HERE=${0:A:h}
LAUNCHER=$HERE/hermes-dev
SRC_REPO=$(git -C "$HERE" rev-parse --show-toplevel) || exit 1
NM_SOURCE=${GUARD_TEST_NM_SOURCE:-/Users/justin/Documents/Projects/Business/hermes-cntrl/.worktrees/cntrl-hermes-worker}
TMP=$(mktemp -d "${TMPDIR:-/tmp}/hermes-dev-guard-test.XXXXXX")
typeset -a WORKTREES=()

cleanup() {
  local wt link
  for wt in $WORKTREES; do
    for link in node_modules apps/desktop/node_modules ui-tui/node_modules; do
      [[ -L "$wt/$link" ]] && rm "$wt/$link"
    done
    git -C "$SRC_REPO" worktree remove --force "$wt" >/dev/null 2>&1 || rm -rf "$wt"
  done
  git -C "$SRC_REPO" worktree prune
  rm -rf "$TMP"
  print "harness: removed ${#WORKTREES} temp worktrees and $TMP"
}
trap cleanup EXIT INT TERM

make_worktree() {  # <sha> -> SHIP_WORKTREE (runs in this shell so cleanup sees it)
  local sha=$1 wt=/tmp/guard-proof-$1 link
  git -C "$SRC_REPO" worktree add --detach --quiet "$wt" "$sha" >&2 || return 1
  WORKTREES+=("$wt")
  for link in node_modules apps/desktop/node_modules ui-tui/node_modules; do
    [[ -d "$NM_SOURCE/$link" && ! -e "$wt/$link" ]] && ln -s "$NM_SOURCE/$link" "$wt/$link"
  done
  SHIP_WORKTREE=$wt
}

# Stub gh: `gh api repos/<o>/<r>/commits/<sha>/check-runs?...` -> $GH_STUB_DIR/<sha>.json
mkdir -p "$TMP/bin" "$TMP/gh"
cat >"$TMP/bin/gh" <<'EOF'
#!/bin/sh
[ "$1" = api ] || { echo "stub gh: unsupported $*" >&2; exit 2; }
[ -n "${GH_STUB_OFFLINE:-}" ] && { echo 'error connecting to api.github.com' >&2; exit 1; }
sha=${2#*/commits/}; sha=${sha%%/*}
[ -f "$GH_STUB_DIR/$sha.json" ] && exec cat "$GH_STUB_DIR/$sha.json"
echo 'gh: No commit found for SHA (HTTP 422)' >&2; exit 1
EOF
chmod +x "$TMP/bin/gh"
export PATH="$TMP/bin:$PATH" GH_STUB_DIR="$TMP/gh"

canned() {  # <full sha> <status> <conclusion|null> [name]
  local name=${4:-Desktop launch smoke} conclusion=$3
  [[ "$conclusion" != null ]] && conclusion="\"$conclusion\""
  cat >"$GH_STUB_DIR/$1.json" <<EOF
{"total_count":2,"check_runs":[
 {"id":1,"name":"Desktop typecheck","status":"completed","conclusion":"success","started_at":"2026-09-29T10:00:00Z","html_url":"https://github.com/ThinkBotHQ/hermes-agent/runs/1"},
 {"id":2,"name":"$name","status":"$2","conclusion":$conclusion,"started_at":"2026-09-29T10:00:05Z","html_url":"https://github.com/ThinkBotHQ/hermes-agent/runs/2"}]}
EOF
}

export HERMES_DEV_SOURCE_ONLY=1 HERMES_DEV_GUARD_DIR="$TMP/guard"
source "$LAUNCHER"

failures=0
check() {  # <label> <expect pass|refuse> <reason substring> <T> [base]
  local label=$1 expect=$2 want=$3 T=$4 base=${5:-$4} got started=$SECONDS
  if hermes_dev_guard "$T" "$base"; then got=pass; else got=refuse; fi
  local took=$((SECONDS - started))
  if [[ "$got" == "$expect" && ( -z "$want" || "$GUARD_REASON" == *"$want"* ) ]]; then
    print "CASE OK    $label: $got in ${took}s${GUARD_REASON:+ — $GUARD_REASON}"
  else
    print "CASE WRONG $label: expected $expect${want:+ ($want)}, got $got${GUARD_REASON:+ — $GUARD_REASON}"
    failures=$((failures + 1))
  fi
  [[ "$got" == refuse ]] && guard_refuse "$T" "$base" 2>&1 | sed $'s/\x1b\\[[0-9;]*m//g; s/^/           /'
}

BROKEN_TSC=$(git -C "$SRC_REPO" rev-parse c0deb07865) || exit 1
BROKEN_LAUNCH=$(git -C "$SRC_REPO" rev-parse c0deb0749c) || exit 1
FIXED=$(git -C "$SRC_REPO" rev-parse c0deb0715b) || exit 1

print "harness: node_modules from $NM_SOURCE; guard cache $HERMES_DEV_GUARD_DIR"

# 1. c0deb07865: the build check refuses it before CI is even asked (CI stub says success).
make_worktree "$BROKEN_TSC" || exit 1
canned "$BROKEN_TSC" completed success
check "c0deb07865 typecheck break" refuse "desktop typecheck failed at ${BROKEN_TSC:0:10}" "$BROKEN_TSC"

# 2. c0deb0749c: builds, but the CI launch smoke failed.
make_worktree "$BROKEN_LAUNCH" || exit 1
canned "$BROKEN_LAUNCH" completed failure
check "c0deb0749c launch crash" refuse "launch smoke failure for ${BROKEN_LAUNCH:0:10}" "$BROKEN_LAUNCH"
[[ -f "$HERMES_DEV_GUARD_DIR/$BROKEN_LAUNCH.ok" ]] && print "           (its build check passed and was cached: the refusal is the CI smoke)"

# 3. c0deb0715b: builds and the smoke passed.
make_worktree "$FIXED" || exit 1
canned "$FIXED" completed success
check "c0deb0715b fix" pass "" "$FIXED"

# Cached build + the other refusals, all against c0deb0715b (no rebuild: .ok is cached).
canned "$FIXED" in_progress null
check "c0deb0715b smoke still running" refuse "launch smoke still running for ${FIXED:0:10}" "$FIXED"
export GH_STUB_OFFLINE=1
check "c0deb0715b gh offline" refuse "gh api failed" "$FIXED"
unset GH_STUB_OFFLINE
canned "$FIXED" completed success "Some other check"
check "c0deb0715b no smoke check yet" refuse "no \"Desktop launch smoke\" check" "$FIXED"
canned "$FIXED" completed skipped
check "c0deb0715b smoke skipped, desktop changed since c0deb0749c" refuse "changes desktop files" "$FIXED" "$BROKEN_LAUNCH"
check "c0deb0715b smoke skipped, nothing changed" pass "" "$FIXED" "$FIXED"
canned "$FIXED" completed success
SHIP_WORKTREE=/tmp/guard-proof-$BROKEN_LAUNCH
check "ship worktree not at T" refuse "is at ${BROKEN_LAUNCH:0:10}, not ${FIXED:0:10}" "$FIXED"

# End-to-end: the whole launcher against a throwaway repo, with stub npm (prints instead of
# launching) and the stub gh. The build check is pre-cached so no npm build runs; these cases
# prove where the guard sits relative to the fast-forward, --skip-guard and --prod.
E2E=$TMP/e2e
git init -q -b main "$E2E/repo" && git -C "$E2E/repo" config user.email t@t && git -C "$E2E/repo" config user.name t
mkdir -p "$E2E/repo/apps/desktop" && print a >"$E2E/repo/apps/desktop/x" && git -C "$E2E/repo" add -A && git -C "$E2E/repo" commit -qm A
git -C "$E2E/repo" branch cntrl-hermes-worker
git -C "$E2E/repo" worktree add -q "$E2E/ship" cntrl-hermes-worker
print b >"$E2E/ship/apps/desktop/x" && git -C "$E2E/ship" commit -qam B
E2E_A=$(git -C "$E2E/repo" rev-parse main) E2E_B=$(git -C "$E2E/ship" rev-parse HEAD)
mkdir -p "$E2E/bin" "$E2E/guard" && print cached >"$E2E/guard/$E2E_B.ok"
print '#!/bin/sh\necho "STUB npm $*"' >"$E2E/bin/npm" && chmod +x "$E2E/bin/npm"

launch() {  # <expect head sha> <expect output substring> <label> [args...]
  local want_head=$1 want_out=$2 label=$3; shift 3
  local out head
  out=$(env -u HERMES_DEV_SOURCE_ONLY HERMES_DEV_REPO="$E2E/repo" HERMES_SHIP_WORKTREE="$E2E/ship" \
    HERMES_DEV_GUARD_DIR="$E2E/guard" PATH="$E2E/bin:$PATH" zsh -f "$LAUNCHER" "$@" 2>&1 | sed $'s/\x1b\\[[0-9;]*m//g')
  head=$(git -C "$E2E/repo" rev-parse HEAD)
  if [[ "$head" == "$want_head" && "$out" == *"$want_out"* ]]; then
    print "CASE OK    e2e $label: live HEAD ${head:0:10}"
  else
    print "CASE WRONG e2e $label: live HEAD ${head:0:10} (want ${want_head:0:10}), output lacks: $want_out"
    failures=$((failures + 1))
  fi
  print -r -- "$out" | sed 's/^/           /'
}

# zsh -f: ~/.zshenv must not put the real npm/gh ahead of the stubs (the real npm would launch).
for tool in npm gh; do
  resolved=$(PATH="$E2E/bin:$PATH" zsh -f -c "command -v $tool")
  [[ "$resolved" == "$E2E/bin/$tool" || "$resolved" == "$TMP/bin/$tool" ]] || { print "harness: $tool resolves to $resolved, not the stub; refusing the e2e cases"; exit 1; }
done
cp "$TMP/bin/gh" "$E2E/bin/gh"

canned "$E2E_B" completed failure
launch "$E2E_A" "STUB npm run dev --workspace apps/desktop" "smoke failed: stays on A and still starts"
canned "$E2E_B" in_progress null
launch "$E2E_A" "launch smoke still running for ${E2E_B:0:10}" "smoke running: stays on A" --prod
canned "$E2E_B" completed failure
launch "$E2E_B" "WITHOUT the build check or the CI launch smoke" "--skip-guard: fast-forwards anyway" --skip-guard
git -C "$E2E/repo" reset -q --hard "$E2E_A"
canned "$E2E_B" completed success
launch "$E2E_B" "STUB npm run dev:prod --workspace apps/desktop" "smoke passed: fast-forwards to B" --prod
launch "$E2E_B" "already at cntrl-hermes-worker" "relaunch at B: no guard, no sync"

# The remote ship branch moves ahead of the local one (a push from elsewhere): the launcher must
# say so and must not claim "already at". It stays on the local branch and never moves it.
git init -q --bare "$E2E/remote.git" && git -C "$E2E/repo" remote add myfork "$E2E/remote.git"
print c >"$E2E/ship/apps/desktop/x" && git -C "$E2E/ship" commit -qam C
E2E_C=$(git -C "$E2E/ship" rev-parse HEAD)
git -C "$E2E/ship" push -q myfork cntrl-hermes-worker && git -C "$E2E/ship" reset -q --hard "$E2E_B"
launch "$E2E_B" "myfork/cntrl-hermes-worker is at ${E2E_C:0:10} but the local cntrl-hermes-worker branch is at ${E2E_B:0:10} (1 commits behind)" "remote ahead: says so, stays on B"
launch "$E2E_B" "the remote is ahead at ${E2E_C:0:10}" "remote ahead: the at-local line names the remote sha"
[[ "$(git -C "$E2E/repo" rev-parse cntrl-hermes-worker)" == "$E2E_B" ]] || { print "CASE WRONG e2e remote ahead: the local branch moved"; failures=$((failures + 1)); }

# A remote that accepts the fetch and then stalls must not hold up the launch: a stub git sleeps
# on `fetch` (everything else goes to the real git). The launcher has to give up after its
# time limit, say so, and still start the app on the local branch.
REAL_GIT=$(command -v git)
print "#!/bin/sh\n[ \"\$1\" = fetch ] && { sleep 30; exit 0; }\nexec $REAL_GIT \"\$@\"" >"$E2E/bin/git" && chmod +x "$E2E/bin/git"
export HERMES_SHIP_FETCH_TIMEOUT=1
stall_started=$SECONDS
launch "$E2E_B" "could not fetch myfork/cntrl-hermes-worker within 1s" "stalled fetch: gives up and says so"
launch "$E2E_B" "STUB npm run dev --workspace apps/desktop" "stalled fetch: the app still starts"
stall_took=$((SECONDS - stall_started))
if (( stall_took > 12 )); then print "CASE WRONG e2e stalled fetch: two launches took ${stall_took}s (the fetch was not bounded)"; failures=$((failures + 1)); else print "CASE OK    e2e stalled fetch: two launches took ${stall_took}s"; fi
rm -f "$E2E/bin/git"; unset HERMES_SHIP_FETCH_TIMEOUT

# A git command killed mid-run leaves index.lock behind. The launcher removes it only when it
# is provably stale (over 30 s old AND no git process running). pgrep is stubbed so the cases
# do not depend on what else runs git on this machine.
LOCK=$(git -C "$E2E/repo" rev-parse --git-path index.lock); LOCK="$E2E/repo/$LOCK"
age_lock() { : >"$LOCK"; touch -t "$(date -v-"$1"S +%Y%m%d%H%M.%S)" "$LOCK"; }
lock_case() {  # <label> <expect: gone|kept>
  local state=kept; [[ -e "$LOCK" ]] || state=gone
  if [[ "$state" == "$2" ]]; then print "CASE OK    e2e $1: lock $state"; else print "CASE WRONG e2e $1: lock $state (want $2)"; failures=$((failures + 1)); fi
  rm -f "$LOCK"
}
print '#!/bin/sh\nexit 1' >"$E2E/bin/pgrep" && chmod +x "$E2E/bin/pgrep"   # no git process running
age_lock 90
launch "$E2E_B" "removed a stale git lock" "stale lock, no git running: removed"
lock_case "stale lock, no git running" gone
age_lock 0
launch "$E2E_B" "may be live; not removing it" "fresh lock: left alone"
lock_case "fresh lock" kept
print '#!/bin/sh\nexit 0' >"$E2E/bin/pgrep"                                  # a git process is running
age_lock 90
launch "$E2E_B" "may be live; not removing it" "old lock but git is running: left alone"
lock_case "old lock, git running" kept
rm -f "$E2E/bin/pgrep"

print
if (( failures )); then
  print "harness: $failures case(s) WRONG"
  exit 1
fi
print "harness: all cases OK"
