from splitting import split_onnx_model
from build import build_model
import sys
import os

node_split = sys.argv[1]
os.environ["FINN_BUILD_DIR"] = "/workspace/results/{}/".format(node_split)
print(os.environ["FINN_BUILD_DIR"])
print(node_split)
model_pth = split_onnx_model(node_split)
build_model(model_pth)
