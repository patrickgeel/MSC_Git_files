import onnx
from onnx import helper, TensorProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
import numpy as np
from qonnx.util.basic import gen_finn_dt_tensor
import onnx.version_converter as vc
from onnx.backend.test.case.node import expect

from qonnx.transformation.infer_shapes import InferShapes

class slice_node():
    def __init__(
            self, input_shape = [4], 
            output_shape = [2], param_shape = [1],
            starts_value = [0], ends_value = [2],
            axes_value = [0], splits_value = [],
            input_tensor = "slice_in", output_tensor = "slice_out",
            dtype = TensorProto.INT64, paramdt = "INT64",node_name = "Slice"):
        
        self.name = node_name
        self.inp_shp = [input_shape]
        self.out_shp = [output_shape]
        self.slice_shape = [param_shape]
            
        self.input = helper.make_tensor_value_info(input_tensor, dtype, input_shape)
        self.output = helper.make_tensor_value_info(output_tensor, dtype, output_shape)

        self.slice_attr = {}
        self.slice_attr["starts"] = np.array(starts_value,dtype=np.int64)
        self.slice_attr["ends"] = np.array(ends_value,dtype=np.int64)
        self.slice_attr["axes"] = np.array(axes_value,dtype=np.int64)
        print(splits_value)
        if not splits_value == None:
            self.slice_attr["splits"] = np.array(splits_value,dtype=np.int64)
        
        self.opset_version = helper.make_operatorsetid("", 9)

    def make_node(self):
        ''' 
        Make a new slice node with attributes instead of inputs
        '''
        self.slice_node = helper.make_node(
            "Slice",
            name = self.name,
            inputs=[self.input.name],
            outputs=[self.output.name],
            **self.slice_attr
        )
        return self.slice_node

    def make_model(self,model_name = "Slice_model.onnx"):

        model_config = {}
        model_config["opset_imports"] = [self.opset_version]

        self.model = ModelWrapper(helper.make_model(
            helper.make_graph(
                [self.slice_node],
                inputs=[self.input],
                outputs=[self.output],
                name="slice_graph"
            ), **model_config)
        )
        self.model.transform(InferShapes())
        self.model.save("onnx_model/%s" %model_name)
