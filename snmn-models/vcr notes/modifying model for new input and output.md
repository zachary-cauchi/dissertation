Unlike vqa dataset, no answer vocabulary file is used since questions are multiple-choice.

New GloVE embeddings required converting to `tf.float32`.

Possible solution 1 for output-loading:
* Use an answers file similar to `answers_vqa.txt` containing all valid answers for the questions in order.
* When predicting the answer, only consider those answers included in the answer_choices.

Visualising the model graph:
* `tensorboard --logdir exp_<dataset>/tb/<cfg_name>/`


