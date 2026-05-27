"""Abstract base for any 6-DoF pose producer (SLAM, VIO, sim, replay)."""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable

from ..pose import Pose


PoseCallback = Callable[[Pose], None]


class PoseSource(ABC):
    """Pushes :class:`Pose` samples to a callback in a background thread."""

    def __init__(self) -> None:
        self._cb: PoseCallback | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.fps: float = 0.0

    def start(self, callback: PoseCallback) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("source already running")
        self._cb = callback
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_wrapper, name=type(self).__name__, daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run_wrapper(self) -> None:
        try:
            self._run()
        except Exception as e:                                    # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[{type(self).__name__}] stopped: {e}")

    @abstractmethod
    def _run(self) -> None: ...

    def _emit(self, pose: Pose) -> None:
        if self._cb is not None:
            self._cb(pose)
