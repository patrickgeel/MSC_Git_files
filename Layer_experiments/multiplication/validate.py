import onnxruntime as ort
import numpy as np
from onnx import load

# Load in the onnx model
onnx_model = load("multiplication\onnx_models\multiply_model_split.onnx")

# Set input tensors to random values
in1 = np.random.random([1,3,512,512]).astype(np.float32)
b = np.random.random([1]).astype(np.float32)
# out_ort = np.random.random([1,16,512,512]).astype(np.float32)
output = in1*b

# Set the input dict
input_dict = {"278": in1, "279": b}
sess = ort.InferenceSession(onnx_model.SerializeToString())
out_ort = sess.run(None,input_dict)

# Verify that the result is correct 
print("Result is:", np.allclose(out_ort,output))