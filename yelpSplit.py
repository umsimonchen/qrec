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
import time
import datetime

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
user = df_review_bus['user_id'].sort_values()
user = pd.unique(user)
user = user.tolist()
user_idx = [i for i in range(len(user))]
user_dict = dict(zip(user,user_idx))

business = df_review_bus['business_id'].sort_values()
business = pd.unique(business)
business = business.tolist()
business_idx = [i for i in range(len(business))]
business_dict = dict(zip(business,business_idx))

df_review_bus = df_review_bus.sort_values(by=['user_id', 'business_id'])
j = 0
with open('ratings.txt', 'w') as f:
    for i, row in df_review_bus.iterrows():
        tmsm = time.mktime(datetime.datetime.strptime(row['date'],"%Y-%m-%d %H:%M:%S").timetuple())
        f.write(str(user_dict[row['user_id']])+' '+str(business_dict[row['business_id']])+' '+str(row['stars'])+' '+str(int(tmsm)))
        f.write('\n')
        print('Finished: %d/6990280' %(j+1))
        j+=1




