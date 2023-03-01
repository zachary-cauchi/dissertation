# Preparing the VCR expert layouts

## Prerequisites
* The latest [Stanford Parser](https://nlp.stanford.edu/software/lex-parser.html#Download) release needs to be downloaded and extracted into the `stanford-parser` directory.

## Instructions
* Execute `python get_questions_answers_rationales.py` to extract every question, answer, and rationale into the `input_sentences` directory.
* Execute the below in parallel to obtain the intermediary parse trees. These will take a lot of time to execute due to the length of the sentences. The final generated layouts can be found in the `exp-layouts` directory for questions, answers, and rationales.
  * `./gen_stanford_trees.sh input_sentences/questions.txt`
  * `./gen_stanford_trees.sh input_sentences/answers.txt`
  * `./gen_stanford_trees.sh input_sentences/rationales.txt`
* Execute `python parse.py` to convert the stanford trees generated above into simplified trees.
* Execute the following command to generate the final ground truth layouts and merge them into the imdbs:
  * `python save_layouts_to_imdb.py ../snmn-models/exp_vcr/data/imdb_r152_7x7/imdb_train.npy`
