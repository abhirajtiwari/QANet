from tqdm import tqdm 
import os
import json 
import pandas as pd 

file_path = './dev-v2.0.json'
j = json.load(open(file_path, "rb"))


# for k in j['data'][0]['paragraphs'][0]['qas'][0]['is_impossible']:
# 	print(k, '\n\n')
# # print(j['data'][0]['title'])

# print(j['data'][0]['paragraphs'][0]['qas'][0]['is_impossible'])


# for k in j['data'][0]['paragraphs'][0]['qas']:
# 	if(k['is_impossible']):
# 		print(k['question'], k['plausible_answers'])
	# print(k['is_impossible'])
print(len(j['data'][0]['paragraphs'][0]['qas']))