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

import finn.builder.build_dataflow as build
from qonnx.core.modelwrapper import ModelWrapper
import finn.builder.build_dataflow_config as build_cfg
from custom_steps import (
    custom_step_tinyyolo_preprocess,
    custom_step_tinyyolo_streamline,
    custom_step_tinyyolo_lower,
    custom_step_tinyyolo_convert_to_hls,
    custom_step_partition
)

# model_name = "tinyyolo-20210831"
model_name = "tinyyolo-20210831_updated"
# model_name = "/shares/bulk/pgeel/FINNv0.8.1_repo/TY_build_KV260/finn/notebooks/MSC_Git_files/TinyYolo/tinyyolo-v0.2/old_update_method/tinyyolo_slice_update"
model_filename = "%s.onnx" % model_name


custom_steps = [
    "step_tidy_up",
    custom_step_tinyyolo_preprocess,
    custom_step_tinyyolo_streamline,
    custom_step_tinyyolo_lower,
    custom_step_tinyyolo_convert_to_hls,
    custom_step_partition,
    # "step_create_dataflow_partition",
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_generate_estimate_reports",
    "step_hls_codegen",
    "step_hls_ipgen",
    "step_set_fifo_depths",
    "step_create_stitched_ip",
    "step_measure_rtlsim_performance",
    "step_synthesize_bitfile",
    "step_make_pynq_driver",
    "step_deployment_package"
]

cfg = build_cfg.DataflowBuildConfig(
#     steps = custom_steps,
    output_dir="build-"+model_name, 
    synth_clk_period_ns = 10.0, 
    auto_fifo_depths = False,
    folding_config_file="tinyyolo-config-v0.2.json",
    board = "KV260_SOM",
    stitched_ip_gen_dcp=True,
    shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
    generate_outputs = [
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.STITCHED_IP,
        build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
        build_cfg.DataflowOutputType.BITFILE,
        build_cfg.DataflowOutputType.PYNQ_DRIVER,
        build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
    ],
    # verification options
    # verify_steps = [
    #     build_cfg.VerificationStepType.STITCHED_IP_RTLSIM,
    # ],
    # verify_input_npy="sample_io/test_image_uint8_nhwc.npy",
    # verify_expected_output_npy="sample_io/test_pred.npy",
    # verify_save_full_context=True,
    # verify_save_rtlsim_waveforms=True,
)

build.build_dataflow_cfg(model_filename, cfg)

