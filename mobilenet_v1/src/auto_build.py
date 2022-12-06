import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import os
from warnings import warn
import shutil
import json


# custom steps for mobilenetv1
from src.custom_steps import (
    step_mobilenet_streamline,
    step_mobilenet_convert_to_hls_layers,
    step_mobilenet_convert_to_hls_layers_separate_th,
    step_mobilenet_lower_convs,
    step_mobilenet_slr_floorplan,
)

def streamline_step(model_file,final_output_dir,folding_config_file):
    #Delete previous run results if exist
    if os.path.exists(final_output_dir):
        shutil.rmtree(final_output_dir)
        print("Previous run results deleted!")

    mobilenet_build_steps =  [
        step_mobilenet_streamline,
    ]

    cfg = build.DataflowBuildConfig(
        output_dir          = final_output_dir,
        steps               = mobilenet_build_steps,
        mvau_wwidth_max     = 80,
        target_fps          = 50,
        synth_clk_period_ns = 10.0,
        folding_config_file = folding_config_file,
        board               = "KV260_SOM",
        shell_flow_type     = build_cfg.ShellFlowType.VIVADO_ZYNQ,
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        ]
    )
    
    build.build_dataflow_cfg(model_file, cfg)

def estimate_report(model_file,final_output_dir,folding_config_file):
    #Delete previous run results if exist
    if os.path.exists(final_output_dir):
        shutil.rmtree(final_output_dir)
        print("Previous run results deleted!")

    mobilenet_build_steps =  [
        step_mobilenet_streamline,
        step_mobilenet_lower_convs,
        step_mobilenet_convert_to_hls_layers_separate_th,
        "step_create_dataflow_partition",
        "step_apply_folding_config",
        "step_generate_estimate_reports",
    ]

    cfg = build.DataflowBuildConfig(
        output_dir          = final_output_dir,
        steps               = mobilenet_build_steps,
        mvau_wwidth_max     = 80,
        target_fps          = 50,
        synth_clk_period_ns = 10.0,
        folding_config_file = folding_config_file,
        board               = "KV260_SOM",
        shell_flow_type     = build_cfg.ShellFlowType.VIVADO_ZYNQ,
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        ]
    )
    
    build.build_dataflow_cfg(model_file, cfg)
    
    
def fits_kv260(resource_report):
    max_usage = {'BRAM_18K': 144, 'LUT': 117120, 'URAM': 64, 'DSP': 1248}
    ok = []
    with open(resource_report, 'r') as f:
        usage = json.load(f)['total']
        for k,v in max_usage.items():
            ok.append(v*1.1>=usage[k])
        print(usage)
    return all(ok)


def bit_build(model_file,final_output_dir,folding_config_file,split_node):

    os.environ["FINN_BUILD_DIR"] = "/workspace/results/KV-260/%s" %(split_node)
    inter_build_dir = os.environ["FINN_BUILD_DIR"]

    if os.path.exists(inter_build_dir):
        shutil.rmtree(inter_build_dir)
        print("Previous intermeditate build deleted!")

    #Delete previous run results if exist
    if os.path.exists(final_output_dir):
        shutil.rmtree(final_output_dir)
        os.remove(final_output_dir)
        print("Previous run results deleted!")
        
    mobilenet_build_steps =  [
        step_mobilenet_streamline,
        step_mobilenet_lower_convs,
        step_mobilenet_convert_to_hls_layers_separate_th,
        "step_create_dataflow_partition",
        "step_apply_folding_config",
        "step_generate_estimate_reports",
        "step_hls_codegen",
        "step_hls_ipgen",
        "step_set_fifo_depths",
        "step_create_stitched_ip",
        "step_synthesize_bitfile",
        "step_make_pynq_driver",
        "step_deployment_package",
    ]

    cfg = build.DataflowBuildConfig(
        output_dir          = final_output_dir,
        steps               = mobilenet_build_steps,
        mvau_wwidth_max     = 80,
        target_fps          = 50,
        synth_clk_period_ns = 10.0,
        folding_config_file = folding_config_file,
        board               = "KV260_SOM",
        shell_flow_type     = build_cfg.ShellFlowType.VIVADO_ZYNQ,
        generate_outputs=[
            build_cfg.DataflowOutputType.BITFILE,
            build_cfg.DataflowOutputType.PYNQ_DRIVER,
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
        ]
    )
    try:    
        build.build_dataflow_cfg(model_file, cfg)        
    except:
        pass
