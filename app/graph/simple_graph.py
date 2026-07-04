from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


NodeFn = Callable[..., Any]


@dataclass
class SimpleStateGraph:
    nodes: dict[str, NodeFn] = field(default_factory=dict)
    edges: dict[str, str] = field(default_factory=dict)
    conditional_edges: dict[str, Callable[[Any], str]] = field(default_factory=dict)
    start: str | None = None

    def add_node(self, name: str, fn: NodeFn) -> None:
        self.nodes[name] = fn

    def add_edge(self, src: str, dst: str) -> None:
        self.edges[src] = dst

    def add_conditional_edges(self, src: str, chooser: Callable[[Any], str]) -> None:
        self.conditional_edges[src] = chooser

    def set_start(self, name: str) -> None:
        self.start = name

    def run(self, state: Any) -> Any:
        current = self.start
        while current:
            state.current_node = current
            state = self.nodes[current](state)
            if current in self.conditional_edges:
                current = self.conditional_edges[current](state)
            else:
                current = self.edges.get(current)
        return state

