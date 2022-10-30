import itertools
import numpy as np
import json
import os

import sys; sys.path.append('../../')  # NOQA
from util import text_processing
from collections import Counter

annotations_dir = '../vcr_dataset/Annotations'
imdb_out_dir = './imdb_r152_7x7'
file_sets = ['train.jsonl']
# file_sets = ['test.jsonl', 'train.jsonl', 'val.jsonl']
file_splits = {}
split_folds = {}
img2qid = {}
qid2que = {}
qid2ans = {}
qid2rat = {}
vocabulary = set()

def extract_folds_from_file_set(file_set):
    with open(os.path.join(annotations_dir, file_set)) as f:
        json_qars = list(f)
    
    qar_count = len(json_qars)
    qar_count_digits = len('%s' % qar_count)
    print(f'Loading {qar_count} Question-Answer-Rationale objects')

    for i, json_qar in enumerate(json_qars):
        if i % 100 == 0:
            print(f'Processing QAR ({i + 1:0{qar_count_digits}}/{qar_count})')

        qar = json.loads(json_qar)

        _, qar_split_id = qar['annot_id'].split('-')
        qar_split_id = int(qar_split_id)
        match_fold = qar['match_fold']

        # Assign ids to respective dictionaries.

        if (match_fold not in split_folds):
            split_folds[match_fold] = []
        if (file_set not in file_splits):
            file_splits[file_set] = []
        if (match_fold not in file_splits[file_set]):
            file_splits[file_set].append(match_fold)

        # Assign answers and rationale sources.
        if ('answer_sources' in qar):
            answer_sources = qar['answer_sources']
            rationale_sources = qar['rationale_sources']

            a_id = answer_sources.index(qar_split_id)
            r_id = rationale_sources.index(qar_split_id)

            if (qar_split_id in qid2que):
                print(f'Warning: {qar_split_id} already defined')

            qid2que[qar_split_id] = qar['question']
            qid2ans[qar_split_id] = qar['answer_choices'][a_id]
            qid2rat[qar_split_id] = qar['rationale_choices'][r_id]

        split_folds[match_fold].append(qar)

for file_set in file_sets:
    print(f'Processing {file_set}')
    extract_folds_from_file_set(file_set)
    print(f'Completed {file_set}')
    print(f'Completed {len(file_splits[file_set])} folds')
    print(f'Loaded {sum(len(split_folds[fold]) for fold in file_splits[file_set])} QAR sets across all folds.')

os.makedirs('./imdb_r152_7x7', exist_ok=True)
# np.save('./imdb_r152_7x7/imdb_train2014.npy', np.array(imdb_train2014))
# np.save('./imdb_r152_7x7/imdb_val2014.npy', np.array(imdb_val2014))
# np.save('./imdb_r152_7x7/imdb_trainval2014.npy', np.array(imdb_train2014+imdb_val2014))
# np.save('./imdb_r152_7x7/imdb_test2015.npy', np.array(imdb_test2015))
# np.save('./imdb_r152_7x7/imdb_test-dev2015.npy', np.array(imdb_test_dev2015))
