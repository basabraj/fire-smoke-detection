#!/usr/bin/env bash
# Quick morning-after check: did last night's 00:00-08:00 training run
# succeed on the Jetson GPU, and what did it score?
set -e
cd "$(dirname "$0")"
PY="/home/conglomerate/Desktop/Fire & Smoke/Home/fsenv/bin/python"

echo "=== last 5 training_runs rows ==="
"$PY" - <<'EOF'
import sqlite3
con = sqlite3.connect("dashboard.db")
cols = ["id", "started_at", "finished_at", "status", "images_used",
        "epochs_completed", "precision", "recall", "map50", "map50_95", "notes"]
rows = con.execute(f"SELECT {','.join(cols)} FROM training_runs ORDER BY id DESC LIMIT 5").fetchall()
for r in rows:
    print(dict(zip(cols, r)))
EOF

echo
echo "=== systemd service status ==="
systemctl --user status fire-smoke-dashboard.service --no-pager -l | head -10

echo
echo "=== any CUDA/exception errors in today's journal ==="
journalctl --user -u fire-smoke-dashboard.service --since "today" 2>/dev/null \
  | grep -iE "error|traceback|failed|exception|CUDA error|no kernel image" || echo "(none found)"
