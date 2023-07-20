import requests
from tqdm import *
from urllib.parse import urlparse
import os
import h5py
import numpy as np

files = [
'bert_da_answer_train.h5',
'bert_da_rationale_train.h5',
'bert_da_answer_val.h5',
'bert_da_rationale_val.h5',
'bert_da_answer_test.h5',
'bert_da_rationale_test.h5'
]

for filename in files:
    with h5py.File(os.path.join('bert_embeddings', filename), 'r') as hf:
        for key in tqdm(list(hf.keys())):
            grp_items = {k: np.array(v, dtype=np.float16) for k, v in hf[str(key)].items()}