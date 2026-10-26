"""Renderer appearance and layout defaults. Edit this file to change the drawing."""

ENGINE = "dot"
GRAPH = {
    "rankdir": "TB", "nodesep": "0.25", "ranksep": "0.5",
    "fontname": "Arial", "fontsize": "11", "labelloc": "t", "splines": "spline",
}
# Whitespace suppresses Graphviz's default tooltip containing only the drawing ID.
NODE = {"shape": "plain", "fontname": "Arial", "fontsize": "10", "tooltip": " "}
EDGE = {"fontname": "Arial", "fontsize": "9", "arrowsize": "0.7", "weight": "1"}
CONTINUATION_WEIGHT = "2"  # A modest preference for each source's longest downstream branch.

# Size-based defaults address measured TinyLlama costs; per-call limits override them.
LARGE_GRAPH_THRESHOLD = 1000
LARGE_GRAPH = {"mclimit": "0.25", "nslimit": "0.25", "nslimit1": "1"}

TEXT_LIMIT = 36  # Per visible line; Constant annotations wrap instead of truncating.
TABLE = {
    "BORDER": "1", "CELLBORDER": "0", "CELLSPACING": "0", "CELLPADDING": "4",
    "STYLE": "ROUNDED",
}
PORT_TABLE = {"BORDER": "0", "CELLBORDER": "0", "CELLSPACING": "0", "CELLPADDING": "4"}
PORT_CONTAINER = {"BORDER": "0"}
DEFAULT_BORDER = "#718096"
DEFAULT_FILL = "#eef2f6"
# One colour per family; add unfamiliar operation types here when useful.
NODE_FAMILIES = {
    "input/output": ("#EDEADE", ("Parameter", "Result")),
    "constant": ("#FFE3A3", ("Constant",)),
    "computation": ("#BFDBFE", ("MatMul", "Convolution", "GroupConvolution", "ScaledDotProductAttention")),
    "arithmetic": ("#DDD6FE", (
        "Add", "Subtract", "Multiply", "Divide", "Power", "Sqrt", "ReduceMean", "ReduceSum",
        "ReduceProd", "Maximum", "Minimum", "Cos", "Sin", "Exp", "Log", "Abs", "Negative",
    )),
    "shape/conversion": ("#BCE8DF", (
        "Convert", "ConvertLike", "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Gather",
        "GatherElements", "ShapeOf", "Broadcast", "Concat", "Split", "VariadicSplit", "Slice",
        "StridedSlice", "Range", "Tile", "Pad",
    )),
    "activation": ("#FED7AA", ("Relu", "ReLU", "Swish", "Sigmoid", "Tanh", "Gelu", "Clamp", "Softmax")),
    "logic": ("#D9E8B4", (
        "Select", "Equal", "NotEqual", "Less", "LessEqual", "Greater", "GreaterEqual",
        "LogicalAnd", "LogicalOr", "LogicalNot", "BitwiseAnd", "BitwiseOr",
    )),
    "state/control": ("#FBCFE0", ("ReadValue", "Assign", "If", "Loop", "TensorIterator")),
}
NODE_FILL = {op: colour for colour, operations in NODE_FAMILIES.values() for op in operations}
BODY = {"color": "#94a3b8", "style": "rounded,dashed"}
OUTPUT_BORDER = "#2563a6"
OUTPUT_FILL = "#e8f1fa"
INTERFACE_EDGES = {
    "mapping": {"color": "#2563a6", "fontcolor": "#2563a6", "style": "dashed"},
    "feedback": {
        "color": "#9333a8", "fontcolor": "#9333a8", "style": "dashed", "constraint": "false",
    },
    "control": {
        "color": "#64748b", "fontcolor": "#64748b", "style": "dotted", "constraint": "false",
    },
}
LAYOUT_EDGE = {"style": "invis"}
