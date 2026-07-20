"""Strict MOTChallenge row parsing used by baseline tooling and tests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MotRow:
    frame: int
    track_id: int
    x: float
    y: float
    width: float
    height: float
    score: float
    world_x: float = -1.0
    world_y: float = -1.0
    world_z: float = -1.0

    @classmethod
    def parse(cls, line: str, *, source: str = "<MOT row>") -> "MotRow":
        values = line.strip().split(",")
        if len(values) != 10:
            raise ValueError(f"{source}: expected 10 columns, found {len(values)}")
        try:
            row = cls(
                frame=int(float(values[0])),
                track_id=int(float(values[1])),
                x=float(values[2]),
                y=float(values[3]),
                width=float(values[4]),
                height=float(values[5]),
                score=float(values[6]),
                world_x=float(values[7]),
                world_y=float(values[8]),
                world_z=float(values[9]),
            )
        except ValueError as error:
            raise ValueError(f"{source}: non-numeric value") from error
        if row.frame < 1 or row.track_id < 1:
            raise ValueError(f"{source}: frame and track ID must be positive")
        if row.width <= 0 or row.height <= 0:
            raise ValueError(f"{source}: width and height must be positive")
        return row

    def format(self) -> str:
        return (
            f"{self.frame},{self.track_id},{self.x:.2f},{self.y:.2f},"
            f"{self.width:.2f},{self.height:.2f},{self.score:.4f},"
            f"{self.world_x:g},{self.world_y:g},{self.world_z:g}"
        )


def parse_mot_text(text: str, *, source: str = "<MOT data>") -> list[MotRow]:
    if not text.strip():
        raise ValueError(f"{source}: result is empty")
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.strip():
            rows.append(MotRow.parse(line, source=f"{source}:{line_number}"))
    if not rows:
        raise ValueError(f"{source}: result is empty")
    return rows
