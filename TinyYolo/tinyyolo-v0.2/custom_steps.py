# Copyright (c) 2022, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from qonnx.core.modelwrapper import ModelWrapper
from finn.builder.build_dataflow_config import DataflowBuildConfig, VerificationStepType
import finn.builder.build_dataflow_steps as build_steps
import numpy as np
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.create_generic_partitions import PartitionFromDict
from qonnx.util.basic import get_by_name
import finn.transformation.streamline.absorb as absorb
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.reorder import MoveLinearPastFork
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.general import RemoveUnusedTensors
from qonnx.transformation.general import GiveUniqueNodeNames
import qonnx.core.data_layout as data_layout
import onnx
import qonnx.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.fpgadataflow.set_exec_mode import SetExecMode
from qonnx.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from qonnx.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from qonnx.transformation.infer_shapes import InferShapes
import qonnx.util.pyverilator as pyv
import os
from qonnx.core.onnx_exec import execute_node
from qonnx.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.custom_op.registry import getCustomOp
from qonnx.core.datatype import DataType
from qonnx.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from qonnx.util.pytorch import ToTensor
import brevitas.onnx as bo
from qonnx.transformation.make_input_chanlast import MakeInputChannelsLast
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from qonnx.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from qonnx.transformation.fpgadataflow.set_exec_mode import SetExecMode

def custom_step_tinyyolo_preprocess(model: ModelWrapper, cfg: DataflowBuildConfig):
    # to be able to feed 8-bit camera input directly into the NN, add the divide-by-255
    # preprocessing (equivalent to ToTensor in PyTorch) at the beginning of the model
    totensor_pyt = ToTensor()
    ishape = (1, 3, 512, 512)
    preproc_filename = cfg.output_dir + "/intermediate_models/preproc.onnx"
    bo.export_finn_onnx(totensor_pyt, ishape, preproc_filename)
    pre_model = ModelWrapper(preproc_filename)
    model = model.transform(MergeONNXModels(pre_model))
    # add transpose node to use channels-last input
    model = model.transform(MakeInputChannelsLast())
    # add input quantization annotation
    global_inp_name = model.graph.input[0].name
    model.set_tensor_datatype(global_inp_name, DataType.UINT8)

    return model

def custom_step_partition(model: ModelWrapper, cfg: DataflowBuildConfig):
    upstream_0 = model.find_upstream("Add_36_out0", lambda x: x.name == "MultiThreshold_0")
    upstream_1 = model.find_upstream("Add_40_out0", lambda x: x.name == "MultiThreshold_0")
    wanted_nodes = []
    unwanted_nodes = []
    for ind, node in enumerate(model.graph.node):
        found_0 = get_by_name(upstream_0, node.name, "name") is not None
        found_1 = get_by_name(upstream_1, node.name, "name") is not None
        if found_0 or found_1:
            wanted_nodes.append(ind)
        else:
            unwanted_nodes.append(ind)
    parent=model.transform(PartitionFromDict(
        partitioning={ 0 : wanted_nodes, 1 : unwanted_nodes }, 
        partition_dir=cfg.output_dir + "/intermediate_models"
    ))
    parent.save(cfg.output_dir + "/intermediate_models/partition_parent.onnx")
    return ModelWrapper(cfg.output_dir + "/intermediate_models/partition_0.onnx")

def custom_step_tinyyolo_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold())
    model = model.transform(Streamline())
    model = model.transform(MoveLinearPastFork())
    model = model.transform(Streamline())

    if VerificationStepType.STREAMLINED_PYTHON in cfg._resolve_verification_steps():
        build_steps.verify_step(model, cfg, "streamlined_python", need_parent=False)
    return model


def custom_step_tinyyolo_lower(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(MakeMaxPoolNHWC())
    model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(absorb.AbsorbTransposeIntoResize())
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = model.transform(InferDataTypes())
    model = model.transform(InferDataLayouts())
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(GiveUniqueNodeNames())

    if "lowered_python" in cfg._resolve_verification_steps():
        build_steps.verify_step(model, cfg, "lowered_python", need_parent=False)
    return model

def custom_step_tinyyolo_convert_to_hls(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(to_hls.InferQuantizedStreamingFCLayer("decoupled"))
    model = model.transform(to_hls.InferStreamingMaxPool())
    model = model.transform(to_hls.InferUpsample())
    model = model.transform(to_hls.InferConvInpGen())
    model = model.transform(to_hls.InferVVAU())
    model = model.transform(to_hls.InferDuplicateStreamsLayer())
    model = model.transform(to_hls.InferThresholdingLayer())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InferDataLayouts())
    model = model.transform(InferDataTypes())

    if "initial_hls" in cfg._resolve_verification_steps():
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
        build_steps.verify_step(model, cfg, "initial_hls", need_parent=False)
    return model
