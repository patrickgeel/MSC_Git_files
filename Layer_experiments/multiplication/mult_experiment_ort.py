import numpy as np
import onnx
import onnxruntime as ort
import time
import matplotlib.pyplot as plt
import csv

class Multiply_experiment_class():
    #TODO: Set the input size to the size of a layer in the TINYYOLO 
    def __init__(self):
        self.in1 = np.random.random([4,4]).astype(np.float32)
        self.in2 = np.random.random([4,4]).astype(np.float32)
        self.ort_runtime=[]
        self.out1 = np.random.random([4,4]).astype(np.float32)
        
    def expected_res_mul(self):
        res = np.multiply(self.in1,self.in2)
        return res

    def test_ort_mul(self, model_name):
        model_load = onnx.load(model_name)
        expected_result = self.expected_res_mul()
        input_dict = {"in1": self.in1, "in2": self.in2}
        sess = ort.InferenceSession(model_load.SerializeToString())
        self.out1 = sess.run(None,input_dict)
        return np.allclose(self.out1,expected_result)
           
    
    def throughput_test(self,interations, filename = "ort_results.csv"):
        for i in range(interations):
            start = time.time()
            self.test_ort_mul("multiply_model.onnx")
            end = time.time()
            self.ort_runtime.append(end-start)       
        self.results_publisher()
        np.savetxt(filename , np.asarray(ort_runtime), delimiter=',')
        return self.ort_runtime
    
    def results_publisher(self):
        print("Interations = %d" %(len(self.ort_runtime)))
        print("Min = %f ms" %(np.mean(self.ort_runtime)*1000))
        print("Max = %f ms" %(np.max(self.ort_runtime)*1000))
        print("Mean = %f ms" %(np.min(self.ort_runtime)*1000))