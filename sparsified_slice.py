import pandas as pd
import scipy.sparse as scm
import numpy as np
import sys
import pickle
# First step of the challenge : make the data size manageable
n_slice = 5
slice_size = np.int(np.floor(26501 / 5, ))
slice_size
data_directory = "/home/hjulienne/data_love/ML_competitions/disease_prediction_from_DNA/data/slices/"

size_slice = 10

5*slice_size
slice = pd.read_csv("./data/Xtrain_challenge_owkin.csv",skiprows=0,
nrows = size_slice, index_col=0)
for i in range(2,n_slice):
    print(i)
    start = i * slice_size
    end = (i+1) * slice_size

    slice = pd.read_csv("./data/Xtrain_challenge_owkin.csv",skiprows=start,
    nrows = end, index_col=0)
    sl_sp = scm.lil_matrix(slice.values, dtype=np.bool_)
    fo = data_directory + "slice"+ str(i) +".p"
    pickle.dump( sl_sp, open(fo, "wb" ) )
