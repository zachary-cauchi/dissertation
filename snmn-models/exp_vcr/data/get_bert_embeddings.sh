#!/bin/bash

urls=(
    "https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_answer_train.h5"
    "https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_rationale_train.h5"
    "https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_answer_val.h5"
    "https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_rationale_val.h5"
    "https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_answer_test.h5"
    "https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_rationale_test.h5"
)

echo "Downloading embeddings"
parallel wget -P bert_embeddings ::: ${urls[@]}