import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import os
import shutil

def build_model(model_file):
    # model_file = "split_model_BatchNormalization_10.onnx"

    final_output_dir = "output_final/{}".format(model_file)

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
        shell_flow_type     = build_cfg.ShellFlowType.VIVADO_ZYNQ,
        generate_outputs=[
            build_cfg.DataflowOutputType.BITFILE,
            build_cfg.DataflowOutputType.PYNQ_DRIVER,
            build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
        ]
    )
    build.build_dataflow_cfg(model_file, cfg)