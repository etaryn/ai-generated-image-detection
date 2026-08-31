#!/usr/bin/env bash
# Show what is in the Slurm queue and how the running job is doing.
#
#   ./jobstatus.sh          queue + last 25 lines of the running job's log
#   ./jobstatus.sh -n 60    ...with 60 lines instead
#   ./jobstatus.sh -f       follow the log live (ctrl-C to stop)
#   ./jobstatus.sh -e       show the .err log instead of the .out log
#   ./jobstatus.sh -H       recent job history (finished jobs, exit states)
#   ./jobstatus.sh -j 1234  pin to a specific job id rather than the running one
#
set -uo pipefail

REPO=/home/b/bing2812/ai-generated-image-detection
LOGS=$REPO/slurm_logs
USER_NAME=${USER:-bing2812}

LINES=25
FOLLOW=0
STREAM=out
HISTORY=0
JOB=""

while getopts "n:feHj:h" opt; do
    case "$opt" in
        n) LINES=$OPTARG ;;
        f) FOLLOW=1 ;;
        e) STREAM=err ;;
        H) HISTORY=1 ;;
        j) JOB=$OPTARG ;;
        h) sed -n '2,10p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "try: $0 -h" >&2; exit 2 ;;
    esac
done

echo "=== queue ($(date '+%H:%M:%S')) ==="
squeue -u "$USER_NAME" -o "%.10i %.16P %.20j %.2t %.11M %.11l %.6D %R"

if [ "$HISTORY" -eq 1 ]; then
    echo
    echo "=== recent jobs (last 3 days) ==="
    sacct -u "$USER_NAME" --starttime=now-3days -X \
        --format=JobID%10,JobName%22,State%12,Elapsed,End
    exit 0
fi

# Default to whatever is running right now; -j overrides.
if [ -z "$JOB" ]; then
    JOB=$(squeue -u "$USER_NAME" -h -t RUNNING -o "%i" | head -1)
fi

if [ -z "$JOB" ]; then
    echo
    echo "Nothing running. Most recent log in $LOGS:"
    ls -t "$LOGS"/*.out 2>/dev/null | head -1
    echo "(./jobstatus.sh -H for finished-job states)"
    exit 0
fi

# Logs are named <something>_<jobid>.out / .err, so the id finds the file.
LOG=$(ls -t "$LOGS"/*_"$JOB"."$STREAM" 2>/dev/null | head -1)
if [ -z "$LOG" ]; then
    echo
    echo "No .$STREAM log found for job $JOB under $LOGS"
    exit 1
fi

echo
squeue -j "$JOB" -h -o "job %i (%j) running %M of %l on %N" 2>/dev/null

if [ "$FOLLOW" -eq 1 ]; then
    echo "=== following $LOG (ctrl-C to stop) ==="
    tail -f "$LOG"
else
    echo "=== last $LINES lines of $(basename "$LOG") ==="
    tail -n "$LINES" "$LOG"
fi
