#!/bin/bash
sn="$1"
#python splitting_ty.py "$1"
cp /workspace/results/"$1"/*_zynq_proj*/*.runs/impl_1/top_wrapper_utilization_placed.rpt output_final/split_model_"$1"/
