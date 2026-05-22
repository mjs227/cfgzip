
import os
import ctypes
import psutil
import signal
import threading
from functools import wraps
from typing import Dict, Set, Generator, Tuple, Any, TypeVar



any0 = TypeVar('any0', bound=Any)
any1 = TypeVar('any1', bound=Any)


class MemoryLimitExceeded(MemoryError):
    pass


def sdict_iter(sdict: Dict[any0, Set[any1]]) -> Generator[Tuple[any0, any1], None, None]:
    for k, v in sdict.items():
        for x in v: yield k, x


def sdict_add(sdict: Dict[any0, Set[any1]], key: any0, value: any1) -> None:
    if key in sdict.keys(): sdict[key].add(value)
    else: sdict.update({key: {value}})


def sdict_remove(sdict: Dict[any0, Set[any1]], key: any0, value: any1) -> None:
    sdict[key].remove(value)
    if len(sdict[key]) == 0: sdict.pop(key)


# multiprocessing=False => async exception; multiprocessing=True => SIGINT
def memory_guard(threshold: float = 0.95, poll_interval: float = 0.25, multiprocessing: bool = False):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            eff_threshold = kwargs.pop('mem_threshold', threshold)
            eff_multiproc = kwargs.pop('is_multiprocessing', multiprocessing)

            if eff_threshold is None:
                return fn(*args, **kwargs)

            target_tid = threading.get_ident()
            stop = threading.Event()
            breach_pct = [None]

            def watcher():
                while not stop.wait(poll_interval):
                    pct = psutil.virtual_memory().percent / 100.0

                    if pct >= eff_threshold:
                        breach_pct[0] = pct

                        if eff_multiproc:
                            os.kill(os.getpid(), signal.SIGINT)
                        else:
                            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                                ctypes.c_ulong(target_tid),
                                ctypes.py_object(MemoryLimitExceeded),
                            )

                        break

            t = threading.Thread(target=watcher, daemon=True)
            t.start()

            try:
                return fn(*args, **kwargs)
            except BaseException as e:
                if breach_pct[0] is not None:
                    raise MemoryLimitExceeded(
                        f"system memory usage {breach_pct[0]} exceeded threshold "
                        f"{eff_threshold} during {fn.__qualname__}()"
                    ) from None

                raise e
            finally:
                stop.set()
                t.join(timeout=1.0)

        return wrapper

    return decorator