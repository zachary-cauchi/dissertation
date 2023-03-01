#!/usr/bin/env python

# Convert the generated expert layouts into a list of module names. These module names will be converted to ground truth tokens for use during training.
# The code is based on https://github.com/ronghanghu/n2nmn/blob/master/exp_vqa/data/convert_new_parse_to_gt_layout.ipynb

import re
import os
import sys
from glob import glob
import numpy as np
from contextlib import ExitStack

if len(sys.argv) < 3:
    print(f'Usage: python {os.path.basename(__file__)} <file_prefix> <path_to_imdb>')
    print(f'Example: python {os.path.basename(__file__)} question_answers ../snmn-models/exp_vcr/data/imdb_r152_7x7/imdb_train.npy')
    sys.exit(1)

filename_prefix = sys.argv[1]
input_filenames = glob(f'exp-layouts/{filename_prefix}.*.txt')
input_id_filenames = glob(f'input-sentences/ids.*.txt')

if len(input_filenames) < 1:
    print(f'No files found matching {filename_prefix}. Exiting...')
    sys.exit(1)
elif len(input_filenames) is not len(input_id_filenames):
    print(f'There is a mismatch in the number of id files versus the number of exp layout files.')
    print(f'{len(input_filenames)} exp layout files but there are only {len(input_id_filenames)} id files.')
    print('Exiting...')
    sys.exit(1)

parse2module_dict = {
    'find': '_Find',
    'relate': '_Transform',
    'and': '_And',
    'is': '_Describe', # All the top modules go to '_Describe'
    'describe': '_Describe'
}

input_filenames = sorted(input_filenames)
input_id_filenames = sorted(input_id_filenames)

imdb_path = sys.argv[2]

# Compile a regex pattern which looks for any of the supported keywords.
# This will be used to replace every keyword in the list of layouts.
replacements = sorted(parse2module_dict, key=len, reverse=True)
pattern = re.compile('|'.join(replacements))

def convertTokensInLineToList(line):
    return [parse2module_dict[match] for match in pattern.findall(line.strip())]

ids = []
layouts = []

# The exit stack allows for opening files outside the 'with' statement while still preserving the auto-close functionality.
with ExitStack() as stack:
    print(f'Loading {len(input_filenames)} files.')
    input_files = [ stack.enter_context(open(filename)) for filename in input_filenames ]
    input_ids = [ stack.enter_context(open(filename)) for filename in input_id_filenames ]

    # Process the lines in the input files into arrays.
    ids = [int(id) for id_file in input_ids for id in id_file]
    layouts = sum([list(map(convertTokensInLineToList, tree_file)) for tree_file in input_files], [])

if len(ids) != len(layouts):
    print(f'There\'s a mismatch in the number of ids to layouts ({len(ids)} ids, {len(layouts)} layouts).')
    print(f'Make sure all the files to be loaded are present, are complete, and are prefixed with \'{filename_prefix}\'.')
    sys.exit(1)

print(f'Id and layout counts match ({len(layouts)}).')
print(f'Loading imdb file {imdb_path}.')
imdb = np.load(imdb_path, allow_pickle=True)

print('Splitting layouts per imdb entry.')
ids = np.unique(ids)
layouts_per_id = len(layouts) // len(ids)

layouts = np.reshape(layouts, [len(ids), layouts_per_id])

for id, layout in zip(ids, layouts):
    imdb[id]['gt_layout_qa_tokens'] = layout.tolist()
    imdb[id].pop('gt_layout_question_tokens', None)
    imdb[id].pop('gt_layout_answer_tokens', None)
    imdb[id].pop('gt_layout_rationale_tokens', None)

print('Splitting complete. Saving...')
np.save(imdb_path, imdb, allow_pickle=True)
