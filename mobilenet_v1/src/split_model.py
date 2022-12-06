from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import get_by_name
from qonnx.transformation.create_generic_partitions import PartitionFromDict
import os


def split_model(split_node,op_type,model_dir):
    '''
    Splits and returns the model at the desired node
    '''
    base_model = os.path.join(model_dir ,"mobilenet_streamline.onnx")
    model = ModelWrapper(base_model)

    up = model.find_upstream(split_node, lambda x: x.name == "Div_0")

    wanted, unwanted = [],[]
    for ind,n in enumerate(model.graph.node):
        
        if get_by_name(up,n.name,"name")is not None:
            wanted.append(ind)
        else:
            unwanted.append(ind)
        
    save_dir = "{}/{}/{}".format(model_dir,op_type,split_node)
    parent = model.transform(PartitionFromDict(partitioning={0:wanted,1:unwanted},partition_dir=save_dir))
    return "{}/partition_0.onnx".format(save_dir)
    