from driver_base import FINNExampleOverlay
from driver import io_shape_dict
import os
import json

import argparse

parser = argparse.ArgumentParser(description='A script that does something')
parser.add_argument('bitfile', type=str, help='The first argument')
parser.add_argument('pwd', type=str, help='The working dir')
args = parser.parse_args()


def throughput():
    bitfile = args.bitfile    
    print(bitfile)
    os.chdir(args.pwd)
    print(os.getcwd())
    platform = "zynq-iodma"
    batch_size = 1
    outputfile = "output.npy"
    runtime_weight_dir = "runtime_weights/"

    # instantiate FINN accelerator driver and pass batchsize and bitfile
    accel = FINNExampleOverlay(
        bitfile_name = bitfile, platform = platform,
        io_shape_dict = io_shape_dict, batch_size = batch_size,
        runtime_weight_dir = runtime_weight_dir
    )

    res = accel.throughput_test()
    with open("hw_metrics.json",'w') as f:
        json.dump(res,f)


if __name__ == "__main__":
    throughput()
