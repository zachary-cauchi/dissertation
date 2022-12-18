#!/usr/bin/env python

# Convert the generated expert layouts into a list of module names. These module names will be converted to ground truth tokens for use during training.
# The code is based on https://github.com/ronghanghu/n2nmn/blob/master/exp_vqa/data/convert_new_parse_to_gt_layout.ipynb

import re
import sys
import numpy as np

layouts_dir = 'exp-layouts/'

parse2module_dict = {
    'find': '_Find',
    'relate': '_Transform',
    'and': '_And',
    'is': '_Describe', # All the top modules go to '_Describe'
    'describe': '_Describe'
}

imdb_path = sys.argv[1]

# Compile a regex pattern which looks for any of the supported keywords.
# This will be used to replace every keyword in the list of layouts.
replacements = sorted(parse2module_dict, key=len, reverse=True)
pattern = re.compile('|'.join(replacements))

def convertTokensInLineToList(line):
    return [parse2module_dict[match] for match in pattern.findall(line.strip())]

with open('input_sentences/ids.txt', "r") as id_file, \
     open(layouts_dir + "questions.txt") as questions_file, \
     open(layouts_dir + "answers.txt") as answers_file, \
     open(layouts_dir + "rationales.txt") as rationales_file:
    
    print(f'Opened ids file {id_file.name}')
    print(f'Opened questions file {questions_file.name}')
    print(f'Opened answers file {answers_file.name}')
    print(f'Opened rationales file {rationales_file.name}')

    print(f'loading imdb file {imdb_path}')
    imdbs = np.load(imdb_path, allow_pickle=True)

    print(f'Converting tokens in expert layouts into lists of SNMN module names.')
    ids = list(map(int, map(str.strip, id_file)))
    qlayouts = list(map(convertTokensInLineToList, questions_file))
    alayouts = list(map(convertTokensInLineToList, answers_file))
    rlayouts = list(map(convertTokensInLineToList, rationales_file))

    print('Packing lists into imdb')
    # First, we need to know how many answers and rationale there are per-question.
    q_count = len(ids)
    ans_per_q = len(alayouts) // q_count
    rat_per_q = len(rlayouts) // q_count

    # Next, reshape the lists so that each n-number of answers/rationale are assigned to their respective question.
    # It is assumed that all answers, rationale, and lists are sorted in order of question appearance.
    alayouts = np.reshape(alayouts, [q_count, ans_per_q]).tolist()
    rlayouts = np.reshape(rlayouts, [q_count, rat_per_q]).tolist()

    # Convert the lists into dictionaries mapped by question id, for lookup when rebuilding the imdb.
    qlayouts = dict(zip(ids, qlayouts))
    alayouts = dict(zip(ids, np.reshape(alayouts, [q_count, ans_per_q]).tolist()))
    rlayouts = dict(zip(ids, np.reshape(rlayouts, [q_count, rat_per_q]).tolist()))

    for imdb in imdbs:
        imdb['gt_layout_question_tokens'] = qlayouts[imdb['question_id']]
        imdb['gt_layout_answer_tokens'] = alayouts[imdb['question_id']]
        imdb['gt_layout_rationale_tokens'] = rlayouts[imdb['question_id']]

    print('Saving new imdb file')
    np.save(imdb_path, imdbs, allow_pickle=True)
