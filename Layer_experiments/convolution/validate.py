from unittest import result
import onnxruntime as ort
import numpy as np
from onnx import load
import time 
# Load in the onnx model
onnx_model = load("multiplication\onnx_models\multiply_model_split.onnx")

# Set input tensors to random values
# TODO: Would be nice to automate this input tensor
in1 = np.random.random([1,3,512,512]).astype(np.float32)

# Set the input dict
input_dict = {"280": in1}

sess = ort.InferenceSession(onnx_model.SerializeToString())
start = time.time()
out_ort = sess.run(None,input_dict)
runtime = time.time()-start
print("Runtime (ms): %0.4f" %(runtime*1000))
