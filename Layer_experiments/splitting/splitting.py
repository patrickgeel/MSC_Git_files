import onnx
from onnx import helper, TensorProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.core.datatype import DataType
from qonnx.util.basic import gen_finn_dt_tensor
import os

# Define a finder_fx
def find_input_node(x):
    return 'x' in x.input

def split_onnx_model(split_node=''):
    if split_node=='':
        print("Please define node to split at!")
        return
    # Open the tinyyolo model
    model = ModelWrapper("../tinyyolo_infershapes.onnx")

    # Get onnx nodes
    nodes = model.graph.node
    # Initialize a tensor
    passed_tensors=[]

    ###################################################################################################
    # TODO: This part of the code can most likely be removed and replaced with the find_upstream part.

    for i,n in enumerate(nodes):
        if n.name != split_node:
            passed_tensors.append(n.input[1])
        else:
            s_node = n        
            break
    init_tens = {}
    for t in passed_tensors:
        init_tens[t] = [model.get_initializer(t),model.get_tensor_datatype(t)]
    start_node = s_node.input[0]
    ###################################################################################################

    # Find nodes upstream of the cut node
    upstream_nodes = model.find_upstream(start_node,find_input_node)

    # Reorder the nodes
    up_n_ordered = []
    for n in reversed(upstream_nodes):
        up_n_ordered.append(n)

    # Get input and output tensor shapes
    ish = model.get_tensor_shape('x')
    osh = model.get_tensor_shape(up_n_ordered[-1].output[0])

    # Make tensor value info for the input and output of the model
    # TODO: Replace make_tensor_value_info with model.get_tensor_value_info
    inputs = helper.make_tensor_value_info('x',TensorProto.FLOAT,ish)
    # inputs = model.get_tensor_valueinfo('x')
    outputs = helper.make_tensor_value_info(up_n_ordered[-1].output[0],TensorProto.FLOAT,osh)
    # TODO: Set to output tensor
    # outputs = model.get_tensor_valueinfo(up_n_ordered[-1].output[0])


    # Make a value info list to include in the graph
    value_info = []
    for t in init_tens.keys():
        value_info.append(
            model.get_tensor_valueinfo(t)
        )

    # Create a graph
    new_graph = helper.make_graph(
        name ="new_graph",
        inputs=[inputs],
        outputs=[outputs],
        value_info=value_info,
        nodes=up_n_ordered
    )

    # Create a new model that only contains the nodes desired
    split_model = ModelWrapper(helper.make_model(new_graph))
    # Set initalizer using the import model
    for t in init_tens.keys():
        split_model.set_initializer(t,model.get_initializer(t))
        
    split_model = split_model.transform(InferShapes())
    split_model = split_model.transform(InferDataTypes())
    if not os.path.exists("model_files"):
        os.mkdir("model_files")
    else:
        model_name = "model_files/split_model_{}.onnx".format(split_node)

    split_model.save(model_name)
    
    print("Split at node {}, and saved to {}".format(split_node,model_name))
    return model_name

    