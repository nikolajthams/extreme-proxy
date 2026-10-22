"""The graphs of the simulations. The treatment is X (T in the paper), the
outcome Y, the confounder U is hidden. The order of the edges fixes the order
in which coefficients are drawn, see `dag.Dag.sample_coefficients`."""

from certification_estimators.dag import Dag

# The double-proxy graph: Z is the treatment-inducing and W the outcome-inducing proxy.
PROXY = Dag.from_edges(
    [("U", "Z"), ("U", "W"), ("U", "X"), ("U", "Y"), ("X", "Y"), ("W", "Y"), ("Z", "X")],
    hidden_nodes=["U"],
)

# The instrumental-variable graph.
IV = Dag.from_edges(
    [("U", "X"), ("U", "Y"), ("X", "Y"), ("Z", "X")],
    hidden_nodes=["U"],
)

# The instrumental-variable graph with the edge U -> Z: Z is an invalid instrument.
INVALID_IV = Dag.from_edges(
    [("U", "Z"), ("U", "X"), ("U", "Y"), ("X", "Y"), ("Z", "X")],
    hidden_nodes=["U"],
)
