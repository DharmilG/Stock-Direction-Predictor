"""Graceful shutdown: finish the current unit, flush, then stop."""

from __future__ import annotations

import signal


class GracefulStop:
    def __init__(self) -> None:
        self.requested = False
        self._original: dict[int, object] = {}
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._original[sig] = signal.signal(sig, self._handle)
            except (ValueError, OSError):
                # Not on the main thread; signal handlers are unavailable.
                continue
        self._installed = bool(self._original)

    def _handle(self, signum, frame) -> None:  # noqa: ANN001
        self.requested = True

    def check(self) -> None:
        if self.requested:
            raise KeyboardInterrupt("stop requested by signal")

    def restore(self) -> None:
        for sig, handler in self._original.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                continue
        self._original.clear()
        self._installed = False


__all__ = ["GracefulStop"]
