from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import get_by_name
from qonnx.transformation.create_generic_partitions import PartitionFromDict


def split_model(split_node):
    model_dir = "/home/pgeel/bulk/FINNv0.8.1_repo/TY_build_KV260/finn/notebooks/MSC_Git_files/mobilenet_v1/models"
    base_model = "{}/mobilenetv1-w4a4_pre_post_tidy.onnx".format(model_dir)

    model = ModelWrapper(base_model)
    # split_node = "BatchNormalization_4_out0"

    up = model.find_upstream(split_node, lambda x: x.name == "Div_0")
    wanted, unwanted = [],[]
    for ind,n in enumerate(model.graph.node):
        
        if get_by_name(up,n.name,"name")is not None:
            wanted.append(ind)
        else:
            unwanted.append(ind)
        
    parent = model.transform(PartitionFromDict(partitioning={0:wanted,1:unwanted},partition_dir="{}/estimate/{}".format(model_dir,split_node)))
    # parent.save("split.onnx")
    return "{}/estimate/{}/partition_0.onnx".format(model_dir,split_node)
    