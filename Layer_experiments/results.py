import os 
import pandas as pd
import numpy as np

result_files = ["addition/results/addition_results.csv","convolution/results/conv_results.csv","multiplication/results/multiply_results.csv"]
df = pd.DataFrame()
for f in result_files:
    name = f.split("/")[-1].split(".")[0]+"(ms)"
    temp = np.genfromtxt(f,delimiter=',')*1000
    df[name]= temp
df.to_csv('concated_arm_results.csv',index=False)
print(df)