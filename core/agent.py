# core/agent.py
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class AgentState:
    """One agent at one timestamp (world coords, metres)."""
    id: int
    easting: float
    northing: float
    yaw_deg: float
    length_m: float
    width_m: float
    speed_mps: float = 0.0  # optional, default to 0.0

    @staticmethod
    def from_row(row: pd.Series) -> "AgentState":
        return AgentState(
            id=int(row["id"]),
            easting=float(row["center_easting"]),
            northing=float(row["center_northing"]),
            yaw_deg=float(row.get("yaw", 0.0)),
            length_m=float(row.get("dimension_length", 4.0) or 4.0),
            width_m=float (row.get("dimension_width",  2.0) or 2.0),
            speed_mps=float(row.get("velocity_magnitude", 0.0) or 0.0),
        )
