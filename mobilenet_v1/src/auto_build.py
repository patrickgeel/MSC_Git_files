import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import os
from warnings import warn
import shutil
from multiprocessing import Process
import json

# custom steps for mobilenetv1
from custom_steps import (
    step_mobilenet_streamline,
    step_mobilenet_convert_to_hls_layers,
    step_mobilenet_convert_to_hls_layers_separate_th,
    step_mobilenet_lower_convs,
    step_mobilenet_slr_floorplan,
    custom_step_partition,
)

def bit_build(model_file,final_output_dir):
# model_file = "/home/pgeel/FINNv0.8.1_repo/build_KV260/finn/notebooks/MSC_Git_files/mobilenet_v1/models/mobilenet_streamline.onnx"
# final_output_dir = "half_baseline_build"

    folding_config_file = "/home/pgeel/FINNv0.8.1_repo/build_KV260/finn/notebooks/MSC_Git_files/mobilenet_v1/folding_config/auto_build_folding.json"
    #Delete previous run results if exist
    if os.path.exists(final_output_dir):
        shutil.rmtree(final_output_dir)
        # os.remove(final_output_dir)
        print("Previous run results deleted!")

    mobilenet_build_steps =  [
    #     step_mobilenet_streamline,
        # custom_step_partition,
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
        target_fps          = 25,
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
        build.build_dataflow_cfg(m, cfg)    
    except:
        pass
    
if __name__ == "__main__":   
    processes = []
    outdir = "testing_auto"
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for f in os.listdir("CPU_testing"):
        if os.path.isdir(os.path.join("CPU_testing", f)):
            model_file = os.path.join("CPU_testing", f,"partition_0.onnx")
            output_dir = os.path.join(outdir,f)
            processes.append(Process(target=bit_build,args=(model_file,output_dir)))
    print(processes)
#     for p in pids:
#         p.start()
#     for p in pids:
#         p.join()
