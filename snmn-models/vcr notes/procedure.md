model and data preparation based on the VQA procedures (`exp_vqa` and `model_vqa`).

Visual features for vcr similarly extracted using the same resnet152 model (similar to in the original r2c publication).

imdb files are composed of the following:
```json
[
    {
        "image_name": "COCO_train2014_000000487025",
        "image_path": "/home/cortesza/Projects/snmn/snmn-models/exp_vcr/coco_dataset/images/train2014/COCO_train2014_000000487025.jpg",
        "image_id": 487025,
        "question_id": 4870250,
        "feature_path": "/home/cortesza/Projects/snmn/snmn-models/exp_vcr/data/resnet152_c5_7x7/train2014/COCO_train2014_000000487025.npy",
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

Textual features required the following work:
* Answer file from VQA task is no longer used (since vcr is multiple-choice only).
* snmn uses `GloVe` (non-contextual) whereas r2c uses `BERT` (contextual).
* vcr dataset uses a very different structure to the vqa dataset. The imdb processing scripts will need to be reworked to support and prepare a new format for these types (see `vcr question-annotation sample breakdown.md`).
* Retraining of GloVe embeddings based on vcr vocabulary (see `vcr question-annotation sample breakdown.md`).
  * Same dimensions are kept (300d vectors x words_in_vocab),
