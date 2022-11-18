import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import os
from warnings import warn
import shutil
from custom_steps import (
    step_resnet18_tidy,
    step_resnet18_streamline,
    step_resnet18_convert_to_hls,
    step_resnet18_set_fifo_depths,
    step_resnet18_slr_floorplan)
import argparse

parser = argparse.ArgumentParser("resnet18")
parser.add_argument("--ram_style",type =str, default = "auto")
parser.add_argument("--fold_file",type =str, default = "KV260_build.json")
args = parser.parse_args()

model_file = "models/resnet18_img_3.onnx"
board = "KV260_SOM"
additional = args.ram_style

if additional != "":
    final_output_dir = "build-{}/{}-{}".format(board,model_file.split('.')[0].split('/')[1],additional)
    os.environ["FINN_BUILD_DIR"] = "/workspace/results/%s/%s-%s" %(board,model_file.split('.')[0].split('/')[1],additional)
    inter_build_dir = os.environ["FINN_BUILD_DIR"]
else:
    final_output_dir = "build-{}/{}".format(board,model_file.split('.')[0].split('/')[1])

    os.environ["FINN_BUILD_DIR"] = "/workspace/results/%s/%s" %(board,model_file.split('.')[0].split('/')[1])
    inter_build_dir = os.environ["FINN_BUILD_DIR"]

if os.path.exists(inter_build_dir):
    shutil.rmtree(inter_build_dir)
    print("Previous intermeditate build deleted!")

#Delete previous run results if exist
if os.path.exists(final_output_dir):
    shutil.rmtree(final_output_dir)
    print("Previous run results deleted!")
    
resnet18_build_steps = [
    step_resnet18_tidy,
    step_resnet18_streamline,
    step_resnet18_convert_to_hls,
    "step_create_dataflow_partition",
    "step_apply_folding_config",
    "step_generate_estimate_reports",
#     "step_hls_codegen",
#     "step_hls_ipgen",
#     step_resnet18_set_fifo_depths,
#     step_resnet18_slr_floorplan,
#     "step_synthesize_bitfile",
#     "step_make_pynq_driver",
#     "step_deployment_package",
]

folding_config_file = "folding_config/%s" %args.fold_file
print(folding_config_file)


cfg = build.DataflowBuildConfig(
    output_dir          = final_output_dir,
    steps               = resnet18_build_steps,
    mvau_wwidth_max     = 80,
    target_fps          = 50,
    synth_clk_period_ns = 10.0,
    folding_config_file = folding_config_file,
    board               = board,
    shell_flow_type     = build_cfg.ShellFlowType.VIVADO_ZYNQ,
    generate_outputs=[
        build_cfg.DataflowOutputType.BITFILE,
        build_cfg.DataflowOutputType.PYNQ_DRIVER,
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
    ]
)
build.build_dataflow_cfg(model_file, cfg)