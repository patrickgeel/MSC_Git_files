import os
import pandas as pd
import argparse
import numpy as np
from onnx import load_model


parser = argparse.ArgumentParser("get_operations_metrics")
parser.add_argument("--op_type", type=str,required=True)
parser.add_argument("--model_dir",type=str, default="../models/")
args = parser.parse_args()

def get_op_counts(
    model_dir,
):
    ops = []
    metrics = {}
    model = load_model(model_dir)
    for n in model.graph.node:
        ops.append(n.op_type)
    operations, counts = np.unique(np.array(ops), return_counts=True)
    op_dict = dict(zip(operations, counts.astype(str)))
    return op_dict

model_dir = "{}/{}/".format(args.model_dir,args.op_type)
partion_dir = os.listdir(model_dir)
metrics_0, metrics_1 = {}, {}

for mf in partion_dir:
    node_split = mf
    dir = os.path.join(model_dir, mf)
    if os.path.isdir(dir):
        # print(dir)
        files = [os.path.join(dir, f) for f in os.listdir(
            dir) if (f.endswith("0.onnx") or f.endswith("1.onnx"))]
        for f in files:
            partition = f.split("\\")[-1].split('.onnx')[0]
            ops = get_op_counts(f)

            if partition == "partition_0":
                metrics_0[node_split] = ops
            elif partition == "partition_1":
                metrics_1[node_split] = ops

def generate_csv_file(metrics:list):
    df_0 = pd.DataFrame(metrics[0])
    df_1 = pd.DataFrame(metrics[1])
    # print(model_dir)
    if not os.path.exists("{}/metrics".format(model_dir)):
        os.makedirs("{}/metrics".format(model_dir))
    df_0.to_csv("{}/metrics/op_count_{}_partition_0.csv".format(model_dir, args.op_type))
    df_1.to_csv("{}/metrics/op_count_{}_partition_1.csv".format(model_dir, args.op_type))
    print("Wrote data to file: ", "{}/metrics/op_count_{}".format(model_dir,args.op_type))

generate_csv_file([metrics_0,metrics_1])