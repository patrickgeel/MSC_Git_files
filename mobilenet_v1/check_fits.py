from src.split_model import split_model
from src.auto_build import bit_build, estimate_report, fits_kv260
import os
import shutil
from qonnx.core.modelwrapper import ModelWrapper

def main():
    model = ModelWrapper("models/mobilenetv1-w4a4_pre_post_tidy.onnx")
    splits = []
    for n in model.graph.node:
        if n.op_type == "Mul":
            splits.append(n.output)

    for sn in splits[0::5]:
        split_node = sn[0] 
        print("--"*20,split_node, "--"*20)
        # Split model to get a new model
        model_file = split_model(split_node)
        print(model_file)

        # Estimate reports
        final_output_dir = "build-{}/estimate/{}".format("KV260",split_node)        
        folding_config_file = "folding_config/auto_build_folding.json"
        estimate_report(model_file,final_output_dir,folding_config_file)

        # Check if estimate fits on KV260, If fits make a bit file
        resource_report = "{}/report/estimate_layer_resources.json".format(final_output_dir)
        if fits_kv260(resource_report):
            try:    
                final_output_dir = "build-{}/fit/{}".format("KV260",split_node)
                if not os.path.exists(final_output_dir):
                    bit_build(model_file,final_output_dir,folding_config_file,split_node)
            except:
                pass
        else:
            print("--"*20,"Does not fit", "--"*20)
            print("\t"*20, split_node)
            return

if __name__ == "__main__":
    main()