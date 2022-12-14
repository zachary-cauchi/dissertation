#!/usr/bin/env python

# Sourced from: https://gist.github.com/ronghanghu/67aeb391f4839611d119c73eba53bc5f
import sys
import numpy as np

base_dir = 'input_sentences/'

with open(base_dir + "ids.txt", "w") as id_file, \
     open(base_dir + "questions.txt", "w") as question_file, \
     open(base_dir + "answers.txt", "w") as answer_file, \
     open("rationales.txt", "w") as rationale_file:
  data = np.load(sys.argv[1], allow_pickle=True)
  ids = [str(elm['question_id']) for elm in data]
  questions = [elm['question_str'] for elm in data]
  answers = [' '.join(answer) for elm in data for answer in elm['all_answers']]
  rationales = [' '.join(rationale) for elm in data for rationale in elm['all_rationales']]
  id_file.writelines('\n'.join(ids).lstrip())
  question_file.writelines('\n'.join(questions).lstrip())
  answer_file.writelines('\n'.join(answers).lstrip())
  rationale_file.writelines('\n'.join(rationales).lstrip())
