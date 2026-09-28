"""How many CPU cores the models may use. Stdlib only, so it runs before numpy
or torch are imported (thread settings only take effect if set before then).

Rule: never take the whole machine. Look at what is free, say so, ask. With no
answer (or no terminal), use the free cores minus 2, so the desktop stays usable.
"""
import os
import sys

KEEP_FREE = 2          # cores always left for the UI and other work
THREAD_VARS = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]


def free_cores():
    """(total, busy, free). Per-core usage via psutil if installed, else load average."""
    total = os.cpu_count() or 1
    try:
        import psutil
        per_core = psutil.cpu_percent(interval=0.5, percpu=True)
        busy = sum(p > 50 for p in per_core)
    except ImportError:
        busy = min(total, round(os.getloadavg()[0]))
    return total, busy, max(total - busy, 1)


def choose(requested=None, ask=True):
    total, busy, free = free_cores()
    default = max(1, min(free, total - KEEP_FREE))
    print(f"CPU: {total} cores, about {busy} busy, {free} free. "
          f"Default leaves {KEEP_FREE} for the UI: {default}.")
    n = requested
    if n is None and ask and sys.stdin.isatty():
        ans = input(f"How many cores should the models use? [{default}] ").strip()
        n = int(ans) if ans else None
    n = default if n is None else max(1, min(int(n), total))
    if n > total - KEEP_FREE:
        print(f"note: {n} cores leaves fewer than {KEEP_FREE} free for the UI")
    for v in THREAD_VARS:
        os.environ[v] = str(n)
    print(f"using {n} core(s)")
    return n


def apply_torch(n):
    """Call after importing torch."""
    import torch
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(max(1, min(2, n)))
    except RuntimeError:        # already set in this process
        pass
