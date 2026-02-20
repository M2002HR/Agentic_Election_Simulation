from __future__ import annotations

import sys
import time
from typing import Any


class ProgressBar:
    def __init__(
        self,
        total: int,
        desc: str,
        *,
        width: int = 32,
        logger: Any | None = None,
    ) -> None:
        self.total = max(1, int(total))
        self.desc = desc
        self.width = max(10, int(width))
        self.logger = logger
        self.current = 0
        self.start_ts = time.time()
        self._closed = False
        self._is_tty = sys.stderr.isatty()
        self._last_bucket = -1

        if self._is_tty:
            self._render()
        elif self.logger is not None:
            self.logger.info("%s: %d/%d (0%%)", self.desc, self.current, self.total)

    def __enter__(self) -> "ProgressBar":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def update(self, step: int = 1, detail: str | None = None) -> None:
        if self._closed:
            return
        self.current = min(self.total, self.current + max(0, int(step)))
        if self._is_tty:
            self._render(detail=detail)
        elif self.logger is not None:
            pct = int((self.current / self.total) * 100)
            bucket = pct // 10
            if bucket != self._last_bucket or self.current >= self.total:
                self._last_bucket = bucket
                msg = f"{self.desc}: {self.current}/{self.total} ({pct}%)"
                if detail:
                    msg += f" | {detail}"
                self.logger.info(msg)

    def close(self) -> None:
        if self._closed:
            return
        if self.current < self.total:
            self.current = self.total
            if self._is_tty:
                self._render()
            elif self.logger is not None:
                self.logger.info("%s: %d/%d (100%%)", self.desc, self.current, self.total)
        if self._is_tty:
            sys.stderr.write("\n")
            sys.stderr.flush()
        self._closed = True

    def _render(self, detail: str | None = None) -> None:
        pct = self.current / self.total
        filled = int(round(self.width * pct))
        bar = ("#" * filled) + ("-" * (self.width - filled))
        elapsed = time.time() - self.start_ts
        msg = f"\r{self.desc} [{bar}] {self.current}/{self.total} {pct * 100:5.1f}% | {elapsed:6.1f}s"
        if detail:
            msg += f" | {detail}"
        sys.stderr.write(msg)
        sys.stderr.flush()
