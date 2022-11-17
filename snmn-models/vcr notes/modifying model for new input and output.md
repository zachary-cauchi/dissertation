Unlike vqa dataset, no answer vocabulary file is used since questions are multiple-choice.

New GloVE embeddings required converting to `tf.float32`.

Possible solution 1 for output-loading:
* Use an answers file similar to `answers_vqa.txt` containing all valid answers for the questions in order.
* When predicting the answer, only consider those answers included in the answer_choices.

Visualising the model graph:
* `tensorboard --logdir exp_<dataset>/tb/<cfg_name>/`

Made the model encode the answers in addition to the question and take them as input. This was done by concatenating the answer tokens to the question tokens and feeding them through the rest of the model. This did not improve the accuracy of the model.

Rewrote the input/output of the model to create separate LSTMs per-question/answer and then concatenate them together. Accuracy sporadically is better, but remains inconsistent.

Loss for a given iteration often does not match the accuracy, being higher with higher accuracy or vice versa. The loss function does not converge so the program accuracy never improves.

Problem is one of determining how loss can be calculated for answers without class (since the answer is a multi-token response varying according to choice of answer).

UPDATE:
Changed the model prediction output to equal the number of answers available per-question. The most confident answer is used as selecting the n'th answer.
The benefit of this is that the output aligns with the answer labels, meaning the loss function is able to work correctly with the answer.
Once leaving the program to execute, the loss function eventually converged, reaching a final accuracy of 54%.
