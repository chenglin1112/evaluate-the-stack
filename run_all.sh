#!/usr/bin/env bash
# Recomputes the statistics in the paper from the layer matrices in this repository.
# Every script writes its outputs next to itself. Check them afterwards with:
#   shasum -a 256 -c EXPECTED_OUTPUTS.sha256
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
run() { echo "== $1/$2"; "$PY" "$1/$2.py" > "$1/_run_$2.log" 2>&1 || { echo "   failed, see $1/_run_$2.log"; exit 1; }; }

P="analysis/M4/panel_v2b"   # primary panel (re-collected judges)
A="analysis/M4"             # archived panel
for s in analyze tost stack_replication increment_value phimax extra_stats merge_recollect opus_by_version; do run "$P" "$s"; done
for s in analyze tost stack_replication increment_value phimax extra_stats merge_recollect; do run "$A" "$s"; done
run analysis/M4/recollect_v2b compare_v2a_v2b
echo "done"
