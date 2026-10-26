"""Model paths, synthetic scale input, and benchmark transformation variants."""

from model_graph_viewer.graph import Edge, Graph, Node, TensorMetadata
from model_graph_viewer.transformations import (
    GraphTransformation, HideConstantsTransformation, HighlightTransformation,
    SharedTensorReferencesTransformation,
)


MODELS = {
    "branching": "models/branching/openvino_model.xml",
    "mobilenetv2": "models/mobilenetv2/openvino_model.xml",
    "tinyllama": "models/tinyllama-fp16/openvino_model.xml",
    "gpt2": "models/gpt2/openvino_model.xml",
    "bert-int8": "models/bert-int8/openvino_model.xml",
}

# Increasing simplification; use identical layout defaults for all three views.
DETAIL_LEVELS = {
    "raw": {"show_dtypes": True},
    "baseline": {"show_dtypes": False},
    "improved": {"show_dtypes": False},
}


def synthetic_graph() -> Graph:
    """5001 nodes: one root, 100 branches of 10 five-node diamonds each."""
    tensor = TensorMetadata("fp32", "[?, 64]")
    scalar = TensorMetadata("fp32", "[]")
    nodes = {0: Node(0, "shared root", "Parameter", outputs=[tensor])}
    edges = []
    for branch in range(100):
        previous = 0
        for depth in range(10):
            constant, left, right, merge, output = range(len(nodes), len(nodes) + 5)
            prefix = f"branch {branch} / diamond {depth}"
            nodes[constant] = Node(constant, prefix + " shared scalar", "Constant", outputs=[scalar])
            nodes[left] = Node(left, prefix + " left", "Add", input_count=2, outputs=[tensor])
            nodes[right] = Node(right, prefix + " right", "Multiply", input_count=2, outputs=[tensor])
            nodes[merge] = Node(merge, prefix + " merge", "Add", input_count=2, outputs=[tensor])
            nodes[output] = Node(output, prefix + " output", "Relu", input_count=1, outputs=[tensor])
            edges.extend([
                Edge(previous, 0, left, 0, tensor), Edge(constant, 0, left, 1, scalar),
                Edge(previous, 0, right, 0, tensor), Edge(constant, 0, right, 1, scalar),
                Edge(left, 0, merge, 0, tensor), Edge(right, 0, merge, 1, tensor),
                Edge(merge, 0, output, 0, tensor),
            ])
            previous = output
    return Graph(nodes, edges)


def configured_transformations(variant: str) -> list[GraphTransformation]:
    return {
        "raw": [],
        "baseline": [HideConstantsTransformation()],
        "improved": [HideConstantsTransformation(), SharedTensorReferencesTransformation(8)],
        "highlight-hide": [HighlightTransformation(["branch"]), HideConstantsTransformation()],
    }[variant]
