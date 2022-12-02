from onnx import load_model
import numpy as np

def get_op_counts(
    model_dir,
):
    ops = []
    model = load_model(model_dir)
    for n in model.graph.node:
        ops.append(n.op_type)
    operations, counts = np.unique(np.array(ops), return_counts=True)
    op_dict = dict(zip(operations, counts.astype(str)))
    return op_dict