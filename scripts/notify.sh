#!/usr/bin/env bash
# Send a phone notification through ntfy (https://ntfy.sh) when a command finishes.
#
#   scripts/notify.sh python scripts/05_inject.py      run a command, notify when it ends
#   scripts/notify.sh --message "export checked"        just send a message
#
# The command's output and exit code pass through unchanged, and a failed notification
# never fails the command.
#
# Setup: put your private ntfy topic name in .ntfy_topic at the repo root (gitignored), or
# set NTFY_TOPIC. Anyone who knows the topic can read and send to it, so never commit it.
#
# No internet when it finishes (e.g. a Pi offline)? It retries once a minute for
# NTFY_RETRY_MINUTES (default 120), so the message goes out once the network is back.

set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOPIC="${NTFY_TOPIC:-}"
if [ -z "$TOPIC" ] && [ -f "$REPO_ROOT/.ntfy_topic" ]; then
    TOPIC="$(tr -d '[:space:]' < "$REPO_ROOT/.ntfy_topic")"
fi
case "$(uname -s)" in Darwin) WHERE="Mac" ;; Linux) WHERE="Pi" ;; *) WHERE="$(uname -s)" ;; esac

send() {   # send <title> <message> <tag>
    if [ -z "$TOPIC" ]; then
        echo "notify.sh: no ntfy topic (create .ntfy_topic or set NTFY_TOPIC); not sending" >&2
        return 0
    fi
    local tries=$(( ${NTFY_RETRY_MINUTES:-120} + 1 )) i
    for (( i = 1; i <= tries; i++ )); do
        if curl -sf -m 15 -o /dev/null -H "Title: $1" -H "Tags: $3" -d "$2" "https://ntfy.sh/$TOPIC"; then
            [ "$i" -gt 1 ] && echo "notify.sh: sent after $i tries" >&2
            return 0
        fi
        [ "$i" -eq 1 ] && echo "notify.sh: could not reach ntfy.sh; retrying every minute" >&2
        [ "$i" -lt "$tries" ] && sleep 60
    done
    echo "notify.sh: gave up after $tries tries" >&2
}

if [ "${1:-}" = "--message" ]; then
    send "lm-sdc ($WHERE)" "${2:-done}" "bell"
    exit 0
fi
if [ $# -eq 0 ]; then
    echo "usage: scripts/notify.sh <command...>  |  scripts/notify.sh --message <text>" >&2
    exit 2
fi

start=$(date +%s)
"$@"
status=$?
minutes=$(( ($(date +%s) - start + 30) / 60 ))
name="$*"
[ ${#name} -gt 80 ] && name="${name:0:77}..."
if [ $status -eq 0 ]; then
    send "Done on $WHERE" "$name finished in ${minutes} min" "white_check_mark"
else
    send "FAILED on $WHERE" "$name exited with code $status after ${minutes} min" "x"
fi
exit $status
