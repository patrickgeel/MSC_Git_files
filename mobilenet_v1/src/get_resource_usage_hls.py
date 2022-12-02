import os
import pandas as pd
import json
import argparse

parser = argparse.ArgumentParser("get_resource_usage_hls")
parser.add_argument("--op_type", type=str,required=True)
parser.add_argument("--build_dir",type=str, default="../build-KV260/")

args = parser.parse_args()


def main():
    total = {}
    op_type = args.op_type
    build_dir = args.build_dir
    print(op_type, build_dir)
    split_nodes = [sn for sn in os.listdir(os.path.join(build_dir,op_type)) if os.path.isdir(os.path.join(build_dir,op_type,sn))]


    for split_node in split_nodes:
        with open(f"{build_dir}/{op_type}/{split_node}/report/estimate_layer_resources_hls.json",'r') as fp:
            raw = json.load(fp)
            for v in raw.values():
                temp = dict.fromkeys(v.keys(),0)
                for k in v.keys():
                    temp[k] += int(v[k])
            total[split_node] = temp
    resource_df = pd.DataFrame(total)
    save_dir = os.path.join("../models/",op_type,"metrics",f"resource_usage_hls_{op_type}.csv")
    resource_df.to_csv(save_dir)
    print("Wrote data to:", save_dir)

if __name__ == "__main__":
    main()
