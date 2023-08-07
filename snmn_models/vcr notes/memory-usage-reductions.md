# Memory Usage Reductions

Attempts to reduce VRAM usage to allow for increased batch size.

* Undid redundant dimension usage in answer and rationale structures. No difference was made.
* Tried Adding per-sentence token-length control (T_ENCODER per question, answer, and rationale).
  * Due to concatenation of LSTM outputs at input unit, different-shaped outputs cannot be concatenated.
  * Try rework the architecture completely perhaps (See [https://stackoverflow.com/a/66032522](https://stackoverflow.com/a/66032522)).
* Updated imdb_stats file generation to produce the token length counts.
  * Rationales over 100 tokens for instance barely reach a combined 0.01%, so reducing T_ENCODER length to 99 shouldn't affect performance of training.
* Added XLA and AMP support.
* Increased resnet feature size to 8x8 to try activate mixed precision mode.
