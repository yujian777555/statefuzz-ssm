from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def canonical_json_bytes(payload: Any) -> bytes:
    """返回稳定、无多余空白的UTF-8 JSON字节。"""
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (text + "\n").encode("utf-8")


def atomic_write_json(path: Path, payload: Any) -> None:
    """同目录写临时文件，刷盘后原子替换目标。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        # POSIX目录可显式刷盘；Windows不支持以相同方式打开目录，
        # 但os.replace本身仍提供原子替换语义。
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise

