import numpy as np
import os
import onnx
from onnxruntime_extensions import get_library_path, PyOp, onnx_op, PyOrtFunction
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from onnx.helper import make_attribute
from driver_base import FINNExampleOverlay
from driver import io_shape_dict

model = ModelWrapper("dataflow_parent.onnx")
DOMAIN = "ai.onnx.contrib"

for n in model.graph.node:
    if n.domain == "finn.custom_op.fpgadataflow":
        n.domain = DOMAIN
    if n.op_type == "StreamingDataflowPartition":
        n.name = ''
        n.attribute.append(make_attribute("bitfile","finn-accel.bit"))
        
# print(model.get_tensor_datatype(model.graph.input[0].name))
model.save("dataflow_parent_updated.onnx")
print(io_shape_dict)

platform = "zynq-iodma"
batch_size = 1
bitfile = "finn-accel.bit"
outputfile = "output.npy"
runtime_weight_dir = "runtime_weights/"

# instantiate FINN accelerator driver and pass batchsize and bitfile
accel = FINNExampleOverlay(
    bitfile_name = bitfile, platform = platform,
    io_shape_dict = io_shape_dict, batch_size = batch_size,
    runtime_weight_dir = runtime_weight_dir
)

# Implement the CustomOp by decorating a function with onnx_op
@onnx_op(op_type="StreamingDataflowPartition", inputs=[PyOp.dt_float], outputs=[PyOp.dt_float])
def StreamingDataflowPartition(inputs):
    obuf_normal = accel.execute(inputs)
    return obuf_normal.astype(np.float32)

inp = np.load("dog.npy")

model_func = PyOrtFunction.from_model("./dataflow_parent_updated.onnx")
outputs = model_func(inp)
# print(outputs)
