import dataclasses
import enum
from typing import Any, Optional

import networkx as nx
from cvxopt import matrix, spmatrix
from cvxopt.glpk import ilp

from grace.base import GraphAttrs

# options for the GLPK optimiser
OPTIMIZER_OPTIONS = {
    "tm_lim": 10_000,
    "msg_lev": "GLP_MSG_OFF",
}


class HypothesisType(enum.Enum):
    TERMINUS = 0
    LINK = 1


@dataclasses.dataclass
class Hypothesis:
    """A graph hypothesis.

    Parameters
    ----------
    i : int
        The starting vertex/node.
    j : int
        The ending vertex/node.
    rho : float
        The probability assigned to this hypothesis.
    """

    i: Optional[int] = None
    j: Optional[int] = None
    rho: Optional[float] = None

    @property
    def label(self) -> HypothesisType:
        if self.i == self.j:
            return HypothesisType.TERMINUS
        if self.i is None or self.j is None:
            return HypothesisType.TERMINUS
        return HypothesisType.LINK


def _build_matrices(
    hypotheses: list[Hypothesis],
    N: int,
) -> tuple[spmatrix, matrix]:
    """Build the constraints matrix.

    Returns
    -------
    A : spmatrix
    rho : matrix

    Notes
    -----
    https://cvxopt.org/userguide/matrices.html
    """
    n_hypotheses = len(hypotheses)

    # entries = [(idx, h.i, int(h.j + N)) for idx, h in enumerate(hypotheses)]
    # idx, i, j = zip(*entries)
    # rows = idx + idx
    # cols = i + j

    A = spmatrix([], [], [], (2 * N, n_hypotheses), "d")
    rho = matrix(0.0, (n_hypotheses, 1), "d")
    for idx, h in enumerate(hypotheses):
        if h.i is not None:
            A[h.i, idx] = 1
        if h.j is not None:
            A[int(h.j + N), idx] = 1
        # Must be of in-built type float or np.float64:
        rho[idx] = h.rho.astype(float)

    # # some sanity checks while debugging
    # assert len(rows) == len(cols)
    # assert all([isinstance(x, int) for x in rows])
    # assert all([isinstance(x, int) for x in cols])

    # A = spmatrix([1.0]*len(rows), cols, rows, (2 * N, n_hypotheses), "d")
    # rho = matrix([h.rho for h in hypotheses], (n_hypotheses, 1), "d")
    return A, rho


def optimise_graph(
    graph: nx.Graph,
    *,
    options: dict[str, Any] = OPTIMIZER_OPTIONS,
) -> nx.Graph:
    """Optimise a graph to split into objects.

    Parameters
    ----------
    graph : graph
        A networkx graph containing the graph to be optimized. Each node should
        contain an attribute `prob_detection`, and each edge should have an
        attribute `prob_link`. These are used to perform the optimization.
    options : dict
        A dictionary of options to pass to the GLPK optimizer.

    Returns
    -------
    optimised_graph : graph
        The globally optimal graph.

    Notes
    -----
    Creates a set of hypotheses to reason about how to connect nodes in the
    graph.  Consider a graph:

        G = <V, E>

    For each vertex in V we generate a hypothesis that the vertex is a false
    positive detection, i.e. is it part of the object or not. However, perhaps
    a better formulation is that each vertex is either the start or end of an
    object, in which case, false positives can be both.

    For each edge in E we generate a hypothesis that the edge connects two
    vertices within the object, i.e. that the edge is wholly inside the object.

    Each hypothesis has an associated score to accept (`rho`). We solve an ILP
    to determine the optimal set of hypotheses to accept:

    maximize    rho'*x
    subject to  A*x = b
                x are all binary
                b is a constraint that is set to include all detections

    """
    hypotheses: list[Hypothesis] = []

    # the number of detections (or vertices) in the graph
    n_detections = graph.number_of_nodes()

    # build a set of false positive hypotheses
    for i, n_dict in graph.nodes.data():
        hypotheses.append(
            Hypothesis(
                i=i,
                j=None,
                rho=n_dict[GraphAttrs.NODE_PREDICTION].prob_TN,
            )
        )
        hypotheses.append(
            Hypothesis(
                i=None,
                j=i,
                rho=n_dict[GraphAttrs.NODE_PREDICTION].prob_TN,
            )
        )
    # build a set of link hypotheses
    for i, j, e_dict in graph.edges.data():
        hypotheses.append(
            Hypothesis(
                i=i,
                j=j,
                rho=e_dict[GraphAttrs.EDGE_PREDICTION].prob_TP,
            )
        )

    n_hypotheses = len(hypotheses)

    # given the hypotheses, construct the A and rho matrices
    A, rho = _build_matrices(hypotheses, n_detections)

    # now set up the ILP solver
    G = spmatrix([], [], [], (2 * n_detections, n_hypotheses), "d")

    # NOTE: h cannot be a sparse matrix
    h = matrix(0.0, (2 * n_detections, 1), "d")
    Ix = set()  # empty set of x which are integer
    B = set(range(n_hypotheses))  # signifies all are binary in x
    b = matrix(1.0, (2 * n_detections, 1), "d")
    status, x = ilp(-rho, -G, h, A, b, Ix, B, options=options)

    # only return link hypotheses
    solution = [h for i, h in enumerate(hypotheses) if x[i] > 0]
    solution = [h for h in solution if h.label == HypothesisType.LINK]

    # given the solution, make a new graph that contains only the nodes that
    # we've determined to be connected, i.e. parts of objects

    optimized = nx.Graph()

    # NOTE: we are adding all nodes here. We could probably prune this so that
    # it's only the nodes that are in the solution
    for node, node_data in graph.nodes.data():
        optimized.add_node(node, **node_data)

    # This adds the edges which *are* in the optimised solution, and maintains
    # their respective edge attributes in the new, optimised graph:
    for h in solution:
        if h.label == HypothesisType.LINK:
            edge_data = graph[h.i][h.j]
            optimized.add_edge(h.i, h.j, **edge_data)

    return optimized
