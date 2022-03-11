# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 10:55:32 2022

@author: simon
"""


import csv
import json
import sys
import os
import pandas as pd
import numpy as np

# Load review data
# =============================================================================
# json_file_path="yelp_academic_dataset_review.json"
# csv_file_path="yelp_academic_dataset_review.csv"
# 
# with open(json_file_path, 'r', encoding='utf-8') as fin:
#     for line in fin:
#         line_contents = json.loads(line)
#         headers=line_contents.keys()
#         break
#     print(headers)
#     
# with open(csv_file_path, 'w', encoding='utf-8') as fout:
#     writer = csv.DictWriter(fout, headers)
#     writer.writeheader()
#     with open(json_file_path, 'r', encoding='utf-8') as fin:
#         i=0
#         for line in fin:
#             i+=1
#             print("Finished: %d/6990280" %i)
#             line_contents = json.loads(line)
#             writer.writerow(line_contents)
# 
# df_bus = pd.read_csv(csv_file_path)
# df_reduced = df_bus.drop(['review_id', 'text', 'useful','funny','cool'],axis=1)
# df_cleaned = df_reduced.dropna()
# df_cleaned.to_csv(csv_file_path, index=False)
# =============================================================================

# Load user data
# =============================================================================
# json_file_path="yelp_academic_dataset_user.json"
# csv_file_path="yelp_academic_dataset_user.csv"
# 
# with open(json_file_path, 'r', encoding='utf-8') as fin:
#     for line in fin:
#         line_contents = json.loads(line)
#         headers=line_contents.keys()
#         break
#     print(headers)
#     
# with open(csv_file_path, 'w', encoding='utf-8') as fout:
#     writer = csv.DictWriter(fout, headers)
#     writer.writeheader()
#     with open(json_file_path, 'r', encoding='utf-8') as fin:
#         i=0
#         for line in fin:
#             i+=1
#             print("Finished: %d/1987897" %i)
#             line_contents = json.loads(line)
#             writer.writerow(line_contents)
# 
# df_user_bus = pd.read_csv(csv_file_path)
# df_user_reduced = df_user_bus[['user_id', 'friends']]
# df_user_cleaned = df_user_reduced.dropna()
# df_user_cleaned.to_csv(csv_file_path, index=False)
# =============================================================================

#Name user id to idx
csv_file_path="yelp_academic_dataset_review.csv"
df_review_bus = pd.read_csv(csv_file_path)
df_review_bus.sort_values(by=['user_id', 'business_id'])
df_review_bus['u_idx'] = -1
df_review_bus['b_idx'] = -1
user_dict = {}
business_dict = {}
user_idx = 0
business_idx = 0
for i, row in df_review_bus.iterrows():
    if row['user_id'] in user_dict:
        df_review_bus.loc[i,'u_idx'] = user_dict[row['user_id']]
    else:
        user_dict[row['user_id']] = user_idx
        df_review_bus.loc[i,'u_idx'] = user_idx
        user_idx += 1
    if row['business_id'] in business_dict:
        df_review_bus.loc[i,'b_idx'] = business_dict[row['business_id']]
    else:
        business_dict[row['business_id']] = business_idx
        df_review_bus.loc[i,'b_idx'] = business_idx
        business_idx += 1
    print('Finished: %d/6990280' %(i+1))




