# Preparing the VCR expert layouts

## Prerequisites
* The latest [Stanford Parser](https://nlp.stanford.edu/software/lex-parser.html#Download) release needs to be downloaded and extracted into the `stanford-parser` directory.

## Instructions
* Execute `python get_questions_answers_rationales.py` to extract every question, answer, and rationale into the `input_sentences` directory.
* Execute the below in parallel to obtain the intermediary parse trees. These will take a lot of time to execute due to the length of the sentences.
  * `./run_parser.sh input_sentences/questions.txt`
  * `./run_parser.sh input_sentences/answers.txt`
  * `./run_parser.sh input_sentences/rationales.txt`
* Execute `python parse.py` to convert the stanford trees generated above into the final expert layouts to be used by the program.

The final generated layouts can be found in the `exp-layouts` directory for questions, answers, and rationales.
