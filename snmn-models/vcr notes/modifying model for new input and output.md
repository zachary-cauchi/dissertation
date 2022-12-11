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

Update to model outputs:
Changed the model prediction output to equal the number of answers available per-question. The most confident answer is used as selecting the n'th answer.
The benefit of this is that the output aligns with the answer labels, meaning the loss function is able to work correctly with the answer.
Once leaving the program to execute, the loss function eventually converged, reaching a final accuracy of 54%.

Attention masking over answers:
When going over the question, the model uses an attention mask to know the length of the question token sequence and therefore ignore the empty padding in the question token buffer. This however was not applied to the answers so the model was reading all the answer token buffer instead of the actual sequence of tokens. The model was modified to construct these attention masks for the buffer.
Each attention mask for question and answer alike is then joined together to form a final attention mask, covering the final output layer of the input unit before being fed to the controller unit.

```python
# TODO: Insert pseudocode of att_mask construction.
```

Reduction of answer inputs:
As the model stood, it expected an image, a question, and a set of 4 possible answers as input. With those inputs, it would then give a softmax 4-way softmax output over the given answers, with the highest value reflecting the answer it feels is most likely to be correct. This had the problem of confusing the program, since it would train weights over many answers making it practically impossible for it to train and develop a robust model. When testing the model mentioned previously on the `val` subset - while the model achieved 54% on the `train` subset - it only scored 25% accuracy, equal to random-chance selection and therefore equal to 0% success.
The model has now been modified to expect a single image, question, and answer record per-task, and output a single confidence score on whether the answer is correct or not. The loss function has also been modified to a sigmoid function, suitable for binary classification since this has become a true/false classification problem. Each batch of data is split such that each image, question, and single answer combination is present in order of appearance. When the outputs of the model for that batch are being evaluated, the array of answers is reshaped into a 4-way softmax and treated the same as before for accuracy scoring. When trained on the `train` subset, a notable improvement was seen in accuracy and loss convergence:
```
exp: vcr_scratch, iter = 80000
        loss (vqa) = 0.257328, loss (layout) = 0.000000, loss (rec) = 0.000000
        accuracy (cur) = 0.828125, accuracy (avg) = 0.877427
snapshot saved to ./exp_vcr/tfmodel/vcr_scratch/00080000
```
When tested on the `val` subset, it achieved a score of 43%:
```
exp: vcr_scratch, iter = 80000, final accuracy on val = 0.439587 (11664 / 26534)
```

