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
