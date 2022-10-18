from splitting import split_onnx_model
from build import build_model
import sys


node_split = sys.argv[1]
print(node_split)
model_pth = split_onnx_model(node_split)
build_model(model_pth)