# core/frame.py
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from pathlib import Path

from core.agent import AgentState

@dataclass
class Frame:
    """All agents at a single timestamp."""
    timestamp: pd.Timestamp
    agents: list[AgentState]

    @classmethod
    def from_df(cls, timestamp: pd.Timestamp, df: pd.DataFrame) -> "Frame":
        agents = [AgentState.from_row(r) for _, r in df.iterrows()]
        return cls(timestamp=timestamp, agents=agents)

    # handy when we need the raw table back
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(a.__dict__ for a in self.agents).assign(timestamp=self.timestamp)
    
    def get_agent(self, agent_id: int) -> AgentState | None:
        """Get an agent by ID, or None if not found."""
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None
