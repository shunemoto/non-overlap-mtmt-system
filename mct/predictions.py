"""再同定の結果（Global ID の割り当てと対応付けの根拠）の読み書き。

評価に要るのは `assignments` だけだが、`links` はフェーズ3以降で候補の
再現率を測るのに使うので、構造として読めるようにしておく。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Link", "Predictions", "load_predictions", "save_predictions"]


@dataclass
class Link:
    """2トラックを同一人物と判断した根拠。"""

    from_cam: str
    from_track: str
    to_cam: str
    to_track: str
    score: float | None = None
    st: float | None = None          # 時空間の尤度
    app: float | None = None         # 外観の尤度
    kind: str | None = None          # in_range / out_of_range / self_loop
    zone_from: str | None = None     # 退出ゾーン
    zone_to: str | None = None       # 出現ゾーン

    @property
    def from_key(self) -> tuple[str, int]:
        return (self.from_cam, int(self.from_track))

    @property
    def to_key(self) -> tuple[str, int]:
        return (self.to_cam, int(self.to_track))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "from": {"cam": self.from_cam, "track": self.from_track},
            "to": {"cam": self.to_cam, "track": self.to_track},
        }
        # None のキーは落とす。誤り分析に使う列なので、入っていないことと
        # 値が無いことを読み手が区別できるようにする。
        for key in ("score", "st", "app", "kind", "zone_from", "zone_to"):
            value = getattr(self, key)
            if value is not None:
                d[key] = value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Link:
        return cls(
            from_cam=str(d["from"]["cam"]),
            from_track=str(d["from"]["track"]),
            to_cam=str(d["to"]["cam"]),
            to_track=str(d["to"]["track"]),
            score=d.get("score"),
            st=d.get("st"),
            app=d.get("app"),
            kind=d.get("kind"),
            zone_from=d.get("zone_from"),
            zone_to=d.get("zone_to"),
        )


@dataclass
class Predictions:
    date: str
    method: str = ""
    assignments: dict[str, dict[str, str]] = field(default_factory=dict)
    links: list[Link] = field(default_factory=list)
    meta_extra: dict[str, Any] = field(default_factory=dict)
    """実行条件（α や候補数など）。読み書きはするが評価には使わない。"""

    @property
    def n_global_ids(self) -> int:
        return len({g for tracks in self.assignments.values() for g in tracks.values()})

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": {
                "date": self.date,
                "method": self.method,
                **self.meta_extra,
                "n_global_ids": self.n_global_ids,
                "n_links": len(self.links),
            },
            "assignments": self.assignments,
            "links": [link.to_dict() for link in self.links],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Predictions:
        meta = d.get("meta") or {}
        return cls(
            date=str(meta.get("date", "")),
            method=str(meta.get("method", "")),
            meta_extra={
                k: v for k, v in meta.items()
                if k not in ("date", "method", "n_global_ids", "n_links")
            },
            assignments={
                str(cam): {str(t): str(g) for t, g in tracks.items()}
                for cam, tracks in (d.get("assignments") or {}).items()
            },
            links=[Link.from_dict(x) for x in (d.get("links") or [])],
        )


def load_predictions(path: str | Path) -> Predictions:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"予測結果が見つかりません: {p}")
    with p.open("r", encoding="utf-8") as f:
        return Predictions.from_dict(json.load(f))


def save_predictions(predictions: Predictions, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(predictions.to_dict(), f, ensure_ascii=False, separators=(",", ":"))
    return p
