import time
from collections import defaultdict

_mark = None
_times = defaultdict(int)

def start():
    global _mark
    _mark = time.time()

def log(index):
    global _times, _mark
    _times[index] += time.time() - _mark
    _mark = time.time()

def repr():
    max_key = max([len(key) for key in _times])
    return "".join([
        f"{key:<{max_key}} : {val}\n" 
        for key, val in _times.items()
    ])

def write(path, msg=""):
    with open(path, 'a') as f:
        f.write(msg + "\n")
        f.write(repr() + "\n")
