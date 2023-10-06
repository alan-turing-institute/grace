import pyarrow as pa

from grace.base import GraphAttrs


NODE_SCHEMA = pa.schema(
    [
        pa.field(GraphAttrs.NODE_X, pa.float32()),
        pa.field(GraphAttrs.NODE_Y, pa.float32()),
        pa.field(GraphAttrs.NODE_GROUND_TRUTH, pa.int64()),
        pa.field(GraphAttrs.NODE_CONFIDENCE, pa.float32()),
        pa.field(GraphAttrs.NODE_FEATURES, pa.list_(pa.float32())),
    ],
    # metadata={"year": "2023"}
)

EDGE_SCHEMA = pa.schema(
    [
        pa.field(GraphAttrs.EDGE_SOURCE, pa.int64()),
        pa.field(GraphAttrs.EDGE_TARGET, pa.int64()),
        pa.field(GraphAttrs.EDGE_GROUND_TRUTH, pa.int64()),
        pa.field(GraphAttrs.EDGE_PREDICTION, pa.list_(pa.float32())),
        pa.field("edge_properties_keys", pa.list_(pa.string())),
        pa.field("edge_properties_values", pa.list_(pa.float32())),
    ],
    # metadata={"year": "2023"}
)
