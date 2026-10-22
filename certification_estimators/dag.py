"""Structural equation models on a directed acyclic graph, for the simulations.

Every node is its own noise plus a weighted sum of its parents (`Dag`), or of
its parents' squares in the nonlinear design (`NonlinearParetoDag`). Nodes
are sampled in topological order and coefficients drawn in the order of the
edges, so a seeded `numpy.random.Generator` reproduces a sample exactly.

Adapted from the graphical-toolbox package
(https://github.com/nikolajthams/graphical-toolbox).
"""

from __future__ import annotations

from typing import Self

import networkx as nx
import numpy as np
import polars as pl


class Dag:
    """A linear structural equation model V_j = eps_j + sum_k a_jk V_k on a DAG;
    subclasses fix the distribution of the noise through `noise`."""

    def __init__(self, G: nx.DiGraph, hidden_nodes: list[str]):
        assert nx.is_directed_acyclic_graph(G), "G must be a DAG"
        self.G = G
        self.hidden_nodes = hidden_nodes

    @classmethod
    def from_edges(cls, edges: list[tuple[str, str]], hidden_nodes: list[str]) -> Self:
        G = nx.DiGraph()
        # Insertion order fixes the topological order and the edge order, and with
        # them which random draw goes to which node and edge; a set would make that
        # order depend on the per-process string-hash seed.
        G.add_nodes_from(list(dict.fromkeys(n for edge in edges for n in edge)))
        G.add_edges_from(edges)
        return cls(G, hidden_nodes)

    @classmethod
    def from_dag(cls, dag: Dag) -> Self:
        """The same graph under this class's noise distribution."""
        return cls(dag.G.copy(), list(dag.hidden_nodes))

    def set_edge_strength(self, parent: str, child: str, weight: float) -> Self:
        self.G.edges[parent, child]["weight"] = weight
        return self

    def sample_coefficients(self, lower: float, upper: float, rng: np.random.Generator) -> Self:
        """Every edge weight drawn independently from the uniform distribution on
        [lower, upper], in the order of the edges."""
        for parent, child in self.G.edges:
            self.set_edge_strength(parent, child, rng.uniform(lower, upper))
        return self

    def noise(self, rng: np.random.Generator, n: int, **parameters) -> np.ndarray:
        raise NotImplementedError

    def parent_term(self, node: str, parent: str, value: np.ndarray) -> np.ndarray:
        """How a parent enters the assignment of a node: linearly."""
        return value

    def sample(self, n: int, rng: np.random.Generator | None = None, drop_hidden: bool = True,
               **noise_parameters) -> pl.DataFrame:
        rng = np.random.default_rng() if rng is None else rng
        data: dict[str, np.ndarray] = {}
        for node in nx.topological_sort(self.G):
            value = self.noise(rng, n, **noise_parameters)
            for parent in self.G.predecessors(node):
                weight = self.G.edges[parent, node].get("weight", 1.0)
                value = value + weight * self.parent_term(node, parent, data[parent])
            data[node] = value
        return pl.DataFrame(data).drop(self.hidden_nodes if drop_hidden else [])


class GaussianDag(Dag):
    """Standard Gaussian noise."""

    def noise(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.normal(size=n)


class StudentTDag(Dag):
    """Student-t noise with `df` degrees of freedom."""

    def noise(self, rng: np.random.Generator, n: int, df: float) -> np.ndarray:
        return rng.standard_t(df, size=n)


class ParetoDag(Dag):
    """Pareto noise with P(eps > x) = (1 + x)^{-alpha} for x >= 0, alpha the
    `tail_index`, so that every variable is non-negative."""

    def noise(self, rng: np.random.Generator, n: int, tail_index: float) -> np.ndarray:
        return rng.pareto(tail_index, size=n)


class NonlinearParetoDag(ParetoDag):
    """Pareto noise, and every parent enters through its square except the
    treatment in the assignment of the outcome, which stays linear."""

    def __init__(self, G: nx.DiGraph, hidden_nodes: list[str], outcome: str = "Y", treatment: str = "X"):
        super().__init__(G, hidden_nodes)
        self.outcome, self.treatment = outcome, treatment

    def parent_term(self, node: str, parent: str, value: np.ndarray) -> np.ndarray:
        return value if (node == self.outcome and parent == self.treatment) else value ** 2
