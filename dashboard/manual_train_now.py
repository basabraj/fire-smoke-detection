"""
manual_train_now.py — one-off manual kick of run_nightly_training(), for
use whenever you want to train right now instead of waiting for tonight's
cron-triggered window (or when today's automatic attempt died early and
you don't want to wait for the next one). Not wired into the UI on purpose
(scheduler.py's docstring says there's deliberately no manual trigger) —
this is an operator escape hatch run from the terminal, not a permanent
feature.

respect_window=False: unlike the automatic cron run, a manual run trains
all its epochs straight through regardless of the clock — it won't stop
partway just because config.py's window hours have passed. It still
checkpoints every 5 epochs and can be resumed the normal way (see
scheduler._find_resumable_run) if it's interrupted for some other reason
(crash, SSH drop, Ctrl+C).
"""
from app import app
from scheduler import run_nightly_training

if __name__ == "__main__":
    run_nightly_training(app, respect_window=False)
