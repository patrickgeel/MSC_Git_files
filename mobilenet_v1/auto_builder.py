import sys 
sys.path(1, "src/")
from split_model import split_model
from auto_build import bit_build, estimate_report, fits_kv260
from multiprocessing import Process
from qonnx.core.modelwrapper import ModelWrapper
import argparse

parser = argparse.ArgumentParser("auto_builder")
parser.add_argument("--model_dir", type=str , default="/home/pgeel/bulk/FINNv0.8.1_repo/build_KV260/finn/notebooks/MSC_Git_files/mobilenet_v1/models")
parser.add_argument("--model_file", type=str, default="models/mobilenetv1-w4a4_pre_post_tidy.onnx")
parser.add_argument("--split_op_type", type=str, default="Mul")
parser.add_argument("--jumps_per_op", type=int, default=5, help="This will get every x op_type node")
args = parser.parse_args()

def get_processes():
    model = ModelWrapper(args.model_file)
    model_dir = args.model_dir
    
    processes = []
    op_type = args.split_op_type
    print(model_dir)
    splits = []
    for n in model.graph.node:
        if n.op_type == op_type:
            splits.append(n.output)

    for sn in splits[0::args.jumps_per_op]:
        split_node = sn[0] 
        print("--"*20,split_node, "--"*20)
        # Split model to get a new model
        model_file = split_model(split_node,op_type,model_dir)
        print(model_file)

        # Estimate reports
        final_output_dir = "build-{}/{}/{}".format("KV260",op_type,split_node)        
        folding_config_file = "folding_config/auto_build_folding.json"
        estimate_report(model_file,final_output_dir,folding_config_file)

        # Check if estimate fits on KV260, If fits make a bit file
        resource_report = "{}/report/estimate_layer_resources.json".format(final_output_dir)
        if fits_kv260(resource_report):
            final_output_dir = "build-{}/{}/{}".format("KV260",op_type, split_node)
            processes.append(Process(target=bit_build,args=(model_file,final_output_dir,folding_config_file,split_node,)))
        else:
            print("--"*20,"Does not fit", "--"*20)
            print("\t"*20, split_node)
            break
    return processes

if __name__ == "__main__":
    pids = get_processes()
    for p in pids:
        p.start()
    for p in pids:
        p.join()