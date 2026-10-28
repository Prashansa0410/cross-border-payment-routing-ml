"""Stop every prototype process (cross-platform: supports macOS and Linux; skips itself)."""

import os
import signal
import subprocess
import sys

TARGETS = (
    "event_bus.py",
    "prediction_service.py",
    "decision_service.py",
    "gateway_service.py",
    "audit_consumer.py",
    "run_all.py",
)
me = os.getpid()

if sys.platform == "darwin" or not os.path.exists("/proc"):
    try:
        out = subprocess.check_output(["ps", "-ax", "-o", "pid,command"]).decode()
        for line in out.splitlines()[1:]:
            parts = line.strip().split(maxsplit=1)
            if len(parts) < 2:
                continue
            pid_str, cmd = parts[0], parts[1]
            if not pid_str.isdigit():
                continue
            pid = int(pid_str)
            if pid == me:
                continue
            if any(t in cmd for t in TARGETS) and "stop_all" not in cmd:
                try:
                    os.kill(pid, signal.SIGTERM)
                    print("stopped", pid, cmd.split()[-1])
                except ProcessLookupError:
                    pass
    except Exception as e:
        print("Error stopping processes via ps:", e)
else:
    for pid in filter(str.isdigit, os.listdir("/proc")):
        pid = int(pid)
        if pid == me:
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode()
        except Exception:
            continue
        if (
            cmd.strip()
            and "python" in cmd.split()[0]
            and any(t in cmd for t in TARGETS)
            and "stop_all" not in cmd
        ):
            os.kill(pid, signal.SIGTERM)
            print("stopped", pid, cmd.split()[-1])
