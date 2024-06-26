# BERT embeddings

BERT embeddings were downloaded pre-computed from the following URLs:
`https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_answer_train.h5`
`https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_rationale_train.h5`
`https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_answer_val.h5`
`https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_rationale_val.h5`
`https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_answer_test.h5`
`https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_rationale_test.h5`

All files are hosted by the original authors of `R2C`, the original scripts used to generate these files can be found in the `r2c` repo under `data/get_bert_embeddings` and can be generated using either a pre-trained model or a model trained from scratch. Further instructions can be found in `data/get_bert_embeddings/README.md`.

Unlike GLOVE embeddings, which are context-free embeddings, BERT uses context-aware embeddings, meaning each token in the sentence has a unique vector associated which is distinct from other instances of the same token in other sentences or even in the same sentence.

Due to the size of the files, they cannot be processed all at once without encountering memory errors. Would require loading and processing one element at-a-time.

## H5 database file format

BERT embeddings generated by the BERT repo and by r2c use the `h5` file extension which is a binary file similar to `npy` files. Based on the `extract_features.py` script in the r2c which builds the repo, it looks like the following format is used inside the file:
* All token embeddings are 768-dimensional and extracted from the second-to-last layer in the BERT model.
* Embeddings are extracted into two sets: One set with answers, and one set with rationales.
* Each entry in the dataset has a set of answer/rationale embeddings for all 4 answers per-task following the naming scheme `answer_{answer|rationale}`.
* Each entry also has a set of `ctx` embeddings which represent the question. 1 ctx embedding is present per answer/rationale and gives context to the embeddings.
* The ctx embeddings for rationales is the combined embeddings of both the question and correct answer.
    * In the event that no correct answer is given, The number of ctx embeddings and answer rationale embeddings equals the total combinations of question, answers, and rationals (1q x 4a x 4r = 16 total combinations).
    * Therefore the `train` and `val` dataset will have 4 rationale embedding pairs each, whereas `test` will have 16 pairs.
* A unique id scheme is given to each qar task in the files for each ctx/answer set. They seem to be the same order across the answer and rationale database files.
