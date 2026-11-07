from __future__ import annotations

import hashlib
import json
import os
import selectors
import signal
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .ingress_contract import AttachmentView
from .models import JsonValue
from .task_models import RequestControl

MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_OCR_BYTES = 64 * 1024


class OcrReader(Protocol):
    def read(self, attachment: AttachmentView, control: RequestControl) -> str | None: ...


def _file_hash(path: Path, limit: int) -> tuple[str, int]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise OSError("bounded input required")
        hasher = hashlib.sha256()
        size = 0
        for chunk in iter(lambda: handle.read(65536), b""):
            size += len(chunk)
            if size > limit:
                raise OSError("bounded input required")
            hasher.update(chunk)
        after = os.fstat(handle.fileno())
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise OSError("input changed")
    return hasher.hexdigest(), size


def _ocr_text(content: bytes) -> str | None:
    payload: JsonValue = json.loads(content.decode("utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"lines"}:
        return None
    lines = payload["lines"]
    if not isinstance(lines, list) or len(lines) > 512:
        return None
    texts: list[str] = []
    for line in lines:
        if not isinstance(line, dict):
            return None
        candidates = line.get("candidates")
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= 3:
            return None
        first = candidates[0]
        if not isinstance(first, dict):
            return None
        text = first.get("text")
        if not isinstance(text, str) or len(text) > 4096:
            return None
        texts.append(text)
    return "\n".join(texts)


@dataclass(frozen=True, slots=True)
class LocalOcr:
    executable: Path
    sha256: str

    def read(self, attachment: AttachmentView, control: RequestControl) -> str | None:
        try:
            image = Path(attachment.local_path)
            executable = self.executable.resolve()
            if control.expired or not image.is_absolute() or _file_hash(image, MAX_IMAGE_BYTES) != (attachment.sha256, attachment.size) or _file_hash(executable, 64 * 1024 * 1024)[0] != self.sha256:
                return None
            deadline = min(control.deadline, time.monotonic() + 20.0)
            content = bytearray()
            with subprocess.Popen([str(executable), str(image)], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True) as process:
                assert process.stdout is not None and process.stderr is not None
                try:
                    with selectors.DefaultSelector() as selector:
                        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
                        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
                        while selector.get_map():
                            remaining = deadline - time.monotonic()
                            if control.expired or remaining <= 0:
                                return None
                            for key, _event in selector.select(min(0.1, remaining)):
                                chunk = os.read(key.fd, 4096)
                                if not chunk:
                                    selector.unregister(key.fileobj)
                                elif key.data == "stderr" or len(content) + len(chunk) > MAX_OCR_BYTES:
                                    return None
                                else:
                                    content.extend(chunk)
                    returncode = process.wait(timeout=max(0.001, deadline - time.monotonic()))
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                if returncode != 0 or control.expired or _file_hash(image, MAX_IMAGE_BYTES) != (attachment.sha256, attachment.size):
                    return None
            return _ocr_text(bytes(content))
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
