from graphical_toolbox.generate_data.dag import Dag

PROXY = Dag.from_edges(
    [    
        ("U", "Z"),
        ("U", "W"),
        ("U", "X"),
        ("U", "Y"),
        ("X", "Y"),
        ("W", "Y"),
        ("Z", "X"),        
    ], 
    hidden_nodes=["U"]
)

IV = Dag.from_edges(
    [
        ("U", "X"),
        ("U", "Y"),
        ("X", "Y"),
        ("Z", "X"),        
    ],
    hidden_nodes=["U"]
)

INVALID_IV = Dag.from_edges(
    [
        ("U", "Z"),
        ("U", "X"),
        ("U", "Y"),
        ("X", "Y"),
        ("Z", "X"),
    ],
    hidden_nodes=["U"]
)

TWO_CONFOUNDERS = Dag.from_edges(
    [
        ("U1", "X"),
        ("U1", "Y"),
        ("U2", "Z"),
        ("U2", "Y"),
        ("X", "Y"),
        ("Z", "X"),
    ],
    hidden_nodes=["U1", "U2"]
)

TWO_CONFOUNDERS_WITH_PROXY = Dag.from_edges(
    [
        ("U1", "X"),
        ("U1", "Y"),
        ("U2", "Z"),
        ("U2", "W"),
        ("U2", "Y"),
        ("X", "Y"),
        ("W", "Y"),
        ("Z", "X"),
    ],
    hidden_nodes=["U1", "U2"]
)
