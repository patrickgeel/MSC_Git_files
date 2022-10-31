# %%
import onnx
from onnx import helper,TensorProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
import numpy as np
from qonnx.util.basic import gen_finn_dt_tensor
import onnx.version_converter as vc
from onnx.backend.test.case.node import expect

from qonnx.transformation.infer_shapes import InferShapes

model = ModelWrapper("tinyyolo-20210831.onnx")
model.get_initializer("737")


# %%
# Convolutional Node

kernel_size, stride, pad = 1,0,1

depthwise = False
in_feature_dim = 7
in_chn = 16
conv_param_shape = [64,128,1,1]

idt = DataType["UINT8"]

input_shape = [1,128,16,16]
output_shape = [1,64,16,16]

total_pad = 2 * pad

conv_weight_dt = DataType["UINT4"]

conv_config = {}
conv_config["dilations"] = [1, 1]
conv_config["group"] = [1]
conv_config["kernel_shape"] = [kernel_size, kernel_size]
conv_config["pads"] = [pad, pad, pad, pad]
conv_config["strides"] = [stride, stride]

top_in = helper.make_tensor_value_info("top_in", TensorProto.FLOAT, input_shape)
conv_out = helper.make_tensor_value_info("conv_out", TensorProto.FLOAT, output_shape)
value_info = [helper.make_tensor_value_info("c1", TensorProto.FLOAT, conv_param_shape)]

conv_node = helper.make_node(
        "Conv",
        inputs = ["top_in","c1"],
        outputs = ["conv_out"],
        **conv_config)

conv_model = helper.make_model(
                helper.make_graph(
                    [conv_node],
                    inputs = [top_in],
                    outputs = [conv_out],
                    name = "conv_graph"
                )
)

c1 = np.random.random(conv_param_shape)

conv_model = ModelWrapper(conv_model)
conv_model.set_initializer("c1", c1)
conv_model.save("Conv_model.onnx")

# %%
# shape node 
shape_in = helper.make_tensor_value_info("conv_out", TensorProto.FLOAT, [1,64,16,16])
shape_out = helper.make_tensor_value_info("735", TensorProto.INT64,[4])
shape_node = helper.make_node(
            "Shape",
            inputs = ["conv_out"],
            outputs = ["735"])
shape_graph = helper.make_graph(
                [shape_node],
                inputs = [shape_in],
                outputs = [shape_out],
                name = "shape_graph"
                )
shape_model = ModelWrapper(helper.make_model(shape_graph))
shape_model.transform(InferShapes())
shape_model.save("Shape_model.onnx")

# %% [markdown]
# ### Slice node
# 

# %%
input_shape = [4]
output_shape = [2]
param_shape = [1]
idt = DataType["UINT8"]
param_dt = DataType["INT64"]
slice_in = helper.make_tensor_value_info("735", TensorProto.INT64, input_shape)
slice_out = helper.make_tensor_value_info("slice_out", TensorProto.INT64, output_shape)
slice_attr = {}
slice_attr["starts"] = np.array([0],dtype=np.int64)
slice_attr["ends"] = np.array([2],dtype=np.int64)
slice_attr["axes"] = np.array([0],dtype=np.int64)

# value_info = [
#         helper.make_tensor_value_info("starts", TensorProto.INT8, param_shape),
#         helper.make_tensor_value_info("ends",   TensorProto.INT8, param_shape),
#         helper.make_tensor_value_info("axes",   TensorProto.INT8, param_shape),
# ]



slice_node = helper.make_node(
            "Slice",
            inputs  = ["735"],#,"starts","ends","axes"],
            outputs = ["slice_out"],
            **slice_attr
            )

slice_graph = helper.make_graph(
                [slice_node],
                inputs = [slice_in],
                outputs = [slice_out],
                name = "slice_graph"
                )

model_config = {}
model_config["opset_imports"] = [helper.make_operatorsetid("",9)]

slice_model = ModelWrapper(helper.make_model(slice_graph,**model_config))
slice_model.transform(InferShapes())
onnx.checker.check_model(slice_model.model)
slice_model.save("Slice_model.onnx")

# %% [markdown]
# ### Concat node

# %%
concat_in = helper.make_tensor_value_info("slice_out", TensorProto.INT64, [2])
p1 = helper.make_tensor_value_info("839", TensorProto.INT64,[2])
top_out = helper.make_tensor_value_info("top_out", TensorProto.INT64,[4])
concat_config = {}
concat_config["axis"] = np.int64(0)
concat_node = helper.make_node(
            "Concat",
            inputs = ["slice_out","839"],
            outputs = ["top_out"],
            **concat_config)

concat_model = helper.make_model(
    
    helper.make_graph(    
        [concat_node],
        inputs=[concat_in],
        outputs = [top_out],
        name = "concat_graph",
        value_info=[p1])
)

Concat_model = ModelWrapper(concat_model)
Concat_model.set_initializer("839",np.array([10,10]))
Concat_model.transform(InferShapes())
Concat_model.save("Concat_model.onnx")

# %% [markdown]
# ### Create Full graph

# %%
# Create graph
value_info.append(p1)
# print(value_info)
graph = helper.make_graph(
    nodes = [conv_node,shape_node,slice_node,concat_node],
    name = "slice_graph",
    inputs = [top_in],
    outputs = [top_out],
    value_info = value_info,
)

# %% [markdown]
# ### Create full model

# %%
model_config = {}
model_config["opset_imports"] = [helper.make_operatorsetid("",9)]
modelProto = helper.make_model(graph,**model_config)
model = ModelWrapper(modelProto)

model.set_initializer("starts",np.array([0]))
model.set_initializer("ends",np.array([2]))
model.set_initializer("axes",np.array([0]))
model.set_initializer("c1", np.random.random(conv_param_shape))
model.set_initializer("839",np.array([10,10]))

# model.transform(InferShapes())
model.save("slice.onnx")

# %% [markdown]
# ### Finn build

# %%
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import os
import shutil

model_file = "slice.onnx"

final_output_dir = "output_files/slice"

#Delete previous run results if exist
if os.path.exists(final_output_dir):
    shutil.rmtree(final_output_dir)
    print("Previous run results deleted!")

cfg = build.DataflowBuildConfig(
    output_dir          = final_output_dir,
    mvau_wwidth_max     = 80,
    target_fps          = 1000000,
    synth_clk_period_ns = 10.0,
    board               = "KV260_SOM",
#     steps               = "estimate_only_dataflow_steps",
    shell_flow_type     = build_cfg.ShellFlowType.VIVADO_ZYNQ,
    generate_outputs=[
        build_cfg.DataflowOutputType.BITFILE,
        build_cfg.DataflowOutputType.PYNQ_DRIVER,
        build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
    ]
)

# %%
# %%time
build.build_dataflow_cfg(model_file, cfg)


