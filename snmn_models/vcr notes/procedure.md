model and data preparation based on the VQA procedures (`exp_vqa` and `model_vqa`).

Visual features for vcr similarly extracted using the same resnet152 model (similar to in the original r2c publication).

imdb files are composed of the following:
```json
[
    {
        "image_name": "COCO_train2014_000000487025",
        "image_path": "/home/cortesza/Projects/snmn/snmn_models/exp_vcr/coco_dataset/images/train2014/COCO_train2014_000000487025.jpg",
        "image_id": 487025,
        "question_id": 4870250,
        "feature_path": "/home/cortesza/Projects/snmn/snmn_models/exp_vcr/data/resnet152_c5_7x7/train2014/COCO_train2014_000000487025.npy",
        "question_str": "What shape is the bench seat?",
        "question_tokens": ["what", "shape", "is", "the", "bench", "seat", "?"],
        "all_answers": ["oval", "semi circle", "curved", "curved", "double curve", "banana", "curved", "wavy", "twisting", "curved"],
        "valid_answers": ["oval", "curved", "curved", "banana", "curved", "wavy", "curved"],
        "soft_score_inds": [457, 1524, 58, 1926], // Indices of the word in the answers_vqa vocabulary file.
        "soft_score_target": [0.3333333333333333, 1.0, 0.3333333333333333, 0.3333333333333333], // Calculated with the folling: min(1., <answer_count_in_valid_answers> / 3.)
        "gt_layout_tokens": ["_Find", "_Describe"] // Loaded from the gt_layout npy files using question id as key.
    }
]
```

imdb testing files are composed of the following:
```json
[
    {
        "image_name": "COCO_train2014_000000487025",
        "image_path": "/home/cortesza/Projects/snmn/snmn_models/exp_vcr/coco_dataset/images/train2014/COCO_train2014_000000487025.jpg",
        "image_id": 487025,
        "question_id": 4870250,
        "feature_path": "/home/cortesza/Projects/snmn/snmn_models/exp_vcr/data/resnet152_c5_7x7/train2014/COCO_train2014_000000487025.npy",
        "question_str": "What shape is the bench seat?",
        "question_tokens": ["what", "shape", "is", "the", "bench", "seat", "?"]
    }
]
```

Textual features required the following work:
* Answer file from VQA task is no longer used (since vcr is multiple-choice only).
* snmn uses `GloVe` (non-contextual) whereas r2c uses `BERT` (contextual).
* vcr dataset uses a very different structure to the vqa dataset. The imdb processing scripts will need to be reworked to support and prepare a new format for these types (see `vcr question-annotation sample breakdown.md`).
* Retraining of GloVe embeddings based on vcr vocabulary (see `vcr question-annotation sample breakdown.md`).
  * Same dimensions are kept (300d vectors x words_in_vocab),
* The expert policy/ground-truth layout for the VQA dataset was created by:
  * Running the `get_questions` script sourced [here](https://gist.github.com/ronghanghu/67aeb391f4839611d119c73eba53bc5f).
  * Generating an sp file from the questions using the [Stanford Parser](https://nlp.stanford.edu/software/lex-parser.shtml#Download).
  * Running the `parse` script on the generated sp file.

With all QARs separated and processed, all that's left is building the imdbs into a compatible format as with the above sample record.
The below QAR example shows the current data:
```json
{
    "movie": "1054_Harry_Potter_and_the_prisoner_of_azkaban",
    "objects": ["person", "person", "person", "car", "cellphone", "clock"],
    "interesting_scores": [-1, 0],
    "answer_likelihood": "likely",
    "img_fn": "lsmdc_1054_Harry_Potter_and_the_prisoner_of_azkaban/1054_Harry_Potter_and_the_prisoner_of_azkaban_00.01.46.736-00.01.50.168@0.jpg",
    "metadata_fn": "lsmdc_1054_Harry_Potter_and_the_prisoner_of_azkaban/1054_Harry_Potter_and_the_prisoner_of_azkaban_00.01.46.736-00.01.50.168@0.json",
    "answer_orig": "1 is upset and disgusted.",
    "question_orig": "How is 1 feeling?",
    "rationale_orig": "1's expression is twisted in digust.",
    "question": ["how", "is", "person", "feeling", "?"],
    "answer_match_iter": [3, 0, 2, 1],
    "answer_sources": [5697, 0, 11800, 6763],
    "answer_choices": [
        ["person", "is", "feeling", "amused", "."],
        ["person", "is", "upset", "and", "disgusted", "."],
        ["person", "is", "feeling", "very", "scared", "."],
        ["person", "is", "feeling", "uncomfortable", "with", "person", "."]
    ],
    "answer_label": 1,
    "rationale_choices": [
        ["person", "'", "s", "mouth", "has", "wide", "eyes", "and", "an", "open", "mouth", "."],
        ["when", "people", "have", "their", "mouth", "back", "like", "that", "and", "their", "eyebrows", "lowered", "they", "are", "usually", "disgusted", "by", "what", "they", "see", "."],
        ["person", "are", "seated", "at", "a", "dining", "table", "where", "food", "would", "be", "served", "to", "them", ".", "people", "unaccustomed", "to", "odd", "or", "foreign", "dishes", "may", "make", "disgusted", "looks", "at", "the", "thought", "of", "eating", "it", "."],
        ["person", "'", "s", "expression", "is", "twisted", "in", "disgust", "."]
    ],
    "rationale_sources": [2832, 22727, 16144, 0],
    "rationale_match_iter": [2, 1, 3, 0],
    "rationale_label": 3,
    "img_id": "val-0",
    "question_number": 0,
    "annot_id": "val-0",
    "match_fold": "val-0",
    "match_index": 0
}
```
