#!/usr/bin/env python

import sys
import os
import math
import numpy as np

base_dir = 'input-sentences/'
whole_files_dir = base_dir + 'whole-files/'

if len(sys.argv) < 2:
  print('Usage: python get_questions_answers_rationales.py path_to_imdb.npy')
  sys.exit(1)

# Fetch the data accordingly
data = np.load(sys.argv[1], allow_pickle=True)
ids = [str(elm['question_id']) for elm in data]
questions = [elm['question_str'] for elm in data]
answers = [' '.join(answer) for elm in data for answer in elm['all_answers']]
rationales = [' '.join(rationale) for elm in data for rationale in elm['all_rationales']]

# Join the questions and answers.
ans_per_q = len(answers) // len(questions)
joined_ids = np.repeat(ids, ans_per_q)
qa = [' '.join([questions[i // ans_per_q], answer]) for i, answer in enumerate(answers)]

# Shard the joined strings into partitions for faster processing.
shard_count = max(os.cpu_count() - 1, 1) if len(joined_ids) > 4096 else 1
sharded_ids = np.array_split(joined_ids, shard_count)
sharded_qa = np.array_split(qa, shard_count)

for i in range(shard_count):
  with open(f'{base_dir}ids.{i}.txt', 'w') as id_shard_file, \
       open(f'{base_dir}question_answers.{i}.txt', 'w') as qa_shard_file:
    id_shard_file.writelines('\n'.join(sharded_ids[i]).lstrip())
    qa_shard_file.writelines('\n'.join(sharded_qa[i]).lstrip())

with open(whole_files_dir + "ids.txt", "w") as id_file, \
     open(whole_files_dir + "questions.txt", "w") as question_file, \
     open(whole_files_dir + "answers.txt", "w") as answer_file, \
     open(whole_files_dir + "rationales.txt", "w") as rationale_file:
  # Join the files into 
  id_file.writelines('\n'.join(ids).lstrip())
  question_file.writelines('\n'.join(questions).lstrip())
  answer_file.writelines('\n'.join(answers).lstrip())
  rationale_file.writelines('\n'.join(rationales).lstrip())
