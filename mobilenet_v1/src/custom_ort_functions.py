from onnx import helper

def revert_quantAvgPool(model):
    nodes = [n for n in model.graph.node if n.op_type == 'QuantAvgPool2d']
    attrs = [n.attribute for n in model.graph.node if n.op_type == 'QuantAvgPool2d']
    for node,attr in zip(nodes,attrs):
        for a in attr:
            if a.name == "stride":
                s = a.i
            elif a.name == "kernel":
                k = a.i
        update = helper.make_node(
            "AveragePool",
            inputs=[node.input[0]],
            outputs=[node.output[0]],
            kernel_shape=[k],
            strides=[s],
        )

        model.graph.node.remove(node)
        model.graph.node.append(update)
    return model
        
def set_multithreshold_default(model,save_model):
    '''
    Pass a modelproto model and the save file
    '''
    new_attr = [helper.make_attribute("out_scale", 1.0),
                helper.make_attribute("out_bias", 0.0),
                helper.make_attribute("data_layout","NCHW")]

    for n in model.graph.node:
        if n.op_type == "MultiThreshold":
            out_scale,bias,datalayout = False,False,False
            for na in n.attribute:
                if na.name == "out_scale": out_scale = True
                if na.name == "out_bias": bias = True
                if na.name == "data_layout": datlayout = True
            if not out_scale: n.attribute.append(new_attr[0])
            if not bias: n.attribute.append(new_attr[1])
            if not datalayout: n.attribute.append(new_attr[2])

            n.domain = "ai.onnx.contrib"
    model.save(save_model)