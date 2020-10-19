from tqdm import tqdm 
import os
import json 
import pandas as pd 


file_path = './data/train-v2.0.json'
j = json.load(open(file_path, "rb"))
# print(len(j['data']))
b_1 = {"version":"v2.0", "data":1}
b_2 = {"version": "v2.0", "data": 1}
b_1["data"] = j["data"][0:50]
# b_2["data"] = j["data"][50:]
k_1 = open('./data/train-v2.0_1.json', "w")
# k_2 = open('./data/train-v2.0_2.json', "w")
json.dump(b_1, k_1)
# json.dump(b_2, k_2)
