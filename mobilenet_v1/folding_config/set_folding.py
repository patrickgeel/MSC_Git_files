from qonnx.core.modelwrapper import ModelWrapper
import json

def set_folding(current_dict, new_dict_file_name,model,fold_dir="/home/pgeel/bulk/FINNv0.8.1_repo/TY_build_KV260/finn/notebooks/MSC_Git_files/mobilenet_v1/folding_config/"):
    for n in model.graph.node:
        simd = mh = mw = pe = -1
        for a in n.attribute:
            if a.name == "SIMD":
                simd = current_dict[n.name]["SIMD"]
            elif a.name == "MH":
                mh = a.i
            elif a.name == "MW":
                mw = a.i
            elif a.name == "PE":
                pe = current_dict[n.name]["PE"]
        attr = [simd,mh,mw,pe]
        if not -1 in attr:
            if mh%pe != 0:
                for i in range(pe-int(0.5*pe),pe):
                    if (mh%i == 0):
                        print(mh,i)
                        current_dict[n.name]["PE"] = i
                        break
            elif mw%simd != 0:
                for i in range(simd-int(0.5*simd),simd):
                    if (mw%i == 0):
                        print(mw,i)
                        current_dict[n.name]["SIMD"] = i
                        break

            mw%simd
            mw*mh//(pe*simd)
    print(fold_dir+new_dict_file_name)            
    fn = open(fold_dir + new_dict_file_name+".json",'w')
    json.dump(current_dict,fn)       
    fn.close()
    

def main():    
    model = ModelWrapper("../build-KV260_SOM/partition_1/intermediate_models/step_create_dataflow_partition.onnx")
    current_dict = json.load(open("ZCU104_folding.json"))
    new_file = "Partition_1_folding"
    set_folding(current_dict=current_dict,new_dict_file_name=new_file,model=model)
    
if __name__ == "__main__":
    main()