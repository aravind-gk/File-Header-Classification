#!/usr/bin/env python
# coding: utf-8

'''Importing packages and fetching file names'''

import glob
import pandas as pd
import numpy as np
from math import sqrt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

path = glob.glob('data/*.txt')

'''Feature generation¶'''

# Fetching header names and labels
header_info = pd.read_csv('header_information.csv')
header_info['file_name'] = header_info['file_name'].astype(str)
header_info['has_header'] = header_info['has_header'].astype(str)

info = dict()
for index in range(len(header_info)):
    info[header_info.iloc[index]['file_name']] = header_info.iloc[index]['has_header']
    
# fetching first row for each file
first_row = dict()
for index in range(len(header_info)):
    fpath = 'data/' + header_info.iloc[index]['file_name']
    file = open(fpath, 'r', encoding = 'utf8').read()
    frow = file.split('\n')[0]
    first_row[header_info.iloc[index]['file_name']] = frow
    
# feature 1: number of characters in header
num_chars = dict()
for index in range(len(header_info)):
    fpath = 'data/' + header_info.iloc[index]['file_name']
    file = open(fpath, 'r', encoding = 'utf8').read()
    frow = file.split('\n')[0]
    num_chars[header_info.iloc[index]['file_name']] = len(frow)
    
# feature 2: number of cells in header
num_cells = dict()
for index in range(len(header_info)):
    fpath = 'data/' + header_info.iloc[index]['file_name']
    file = open(fpath, 'r', encoding = 'utf8').read()
    frow = file.split('\n')[0]
    num_cells[header_info.iloc[index]['file_name']] = frow.count(',') + 1
    
# feature 3: average cell length of cells in header
avg_cell_len = dict()
for index in range(len(header_info)):
    fpath = header_info.iloc[index]['file_name']
    frow = first_row[fpath]
    avg_cell_len[fpath] = (num_chars[fpath] - num_cells[fpath] + 1) / num_cells[fpath]
    
# feature 4: percentage of numeric characters in header
numeric_percent = dict()
for index in range(len(header_info)):
    fpath = header_info.iloc[index]['file_name']
    frow = first_row[fpath]
    numbers = sum(c.isdigit() for c in frow)
    numeric_percent[fpath] = numbers / (num_chars[fpath] - num_cells[fpath] + 2) * 100 # adding 2 to prevent div-by-0
    
# feature 5: percentage of alphabetical characters in header
alpha_percent = dict()
for index in range(len(header_info)):
    fpath = header_info.iloc[index]['file_name']
    frow = first_row[fpath]
    letters = sum(c.isalpha() for c in frow)
    alpha_percent[fpath] = letters / (num_chars[fpath] - num_cells[fpath] + 2) * 100
    
# feature 6: variance of cell lengths in header
cell_len_variance = dict()
for index in range(len(header_info)):
    fpath = header_info.iloc[index]['file_name']
    frow = first_row[fpath]
    cells = frow.split(',')
    cell_len_avg = np.mean([len(cell) for cell in cells])
    cell_len_var = sum([(len(cell) - cell_len_avg) ** 2 for cell in cells]) / len(cells)
    cell_len_variance[fpath] = cell_len_var
    
# converting features to pandas series
num_chars_s = pd.Series(num_chars, name = 'num_chars')
num_cells_s = pd.Series(num_cells, name = 'num_cells')
avg_cell_len_s = pd.Series(avg_cell_len, name = 'avg_cell_len')
numeric_percent_s = pd.Series(numeric_percent, name = 'numeric_percent')
alpha_percent_s = pd.Series(alpha_percent, name = 'alpha_percent')
cell_len_variance_s = pd.Series(cell_len_variance, name = 'cell_len_variance')

# creating dataframe containing both generated features and target labels
features = pd.concat([num_chars_s, num_cells_s, avg_cell_len_s, numeric_percent_s, alpha_percent_s, cell_len_variance_s], axis = 1)
target = pd.read_csv('header_information.csv')
target['file_name'] = target['file_name'].astype(str)
target['has_header'] = target['has_header'].astype(str)
target = target.set_index('file_name')
target.columns = ['target']
data = pd.concat([features, target], axis = 1)
data['target'] = data['target'].map({'yes': 1, 'no': 0})

'''Training the model'''

# performing train-test split with given ratios
X = data[['num_chars', 'num_cells', 'avg_cell_len', 'numeric_percent', 'alpha_percent', 'cell_len_variance']]
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# using random forest model for training 
clf = RandomForestClassifier(random_state=1)
clf.fit(X_train.to_numpy(), y_train.to_numpy())
y_pred = clf.predict(X_test.to_numpy())

'''Function for extracting the headers from files'''

def extract_header(fname):
    X = [features.loc[fname].to_numpy()]
    y_pred = clf.predict(X)
    frow = first_row[fname]
    
    ans = dict()
    ans['has_header'] = "yes" if y_pred[0] == 1 else "no"
    ans['header'] = frow.split(',') if y_pred[0] == 1 else []
    return ans

# Examples 

# print(extract_header("0b4b6d05-3024-4922-9910-b288923de027.txt"))

# print(extract_header("74b0863c-c23e-46e5-9021-8e43ad799224.txt"))
