import numpy as np
import json
import os

import sys; sys.path.append('../../')  # NOQA
from collections import Counter
from word2number import w2n

annotations_dir = '../vcr_dataset/Annotations'
images_dir = '../vcr_dataset/vcr1images'
corpus_file = './corpus_vcr.txt'
vocabulary_file = './vocabulary_vcr.txt'
answers_file = './answers_%s_vcr.txt'
imdb_out_dir = './imdb_r152_7x7'
resnet_feature_dir = './resnet152_c5_7x7'
file_sets = {
    'train.jsonl': { 'load_answers': True },
    'val.jsonl': { 'load_answers': True },
    'test.jsonl': { 'load_answers': False}
}

file_splits = {}
split_folds = {}
img2qid = {}
qid2que = {}
corpus = []
vocab = Counter()
split_answers = {}

longest_token_sequences = {

}

word_num_checker_dict = {
    **w2n.american_number_system,
    'and': 0,
    'point': 0
}

def open_image_metadata_file(qar):
    metadata_name = qar['metadata_fn']

    with open(os.path.join(images_dir, metadata_name)) as f:
        return json.load(f)

def preprocess_token(token, qar, metadata):
    # Resolve the token if the token is a reference to an object.
    if isinstance(token, list):
        resolved_token = metadata['names'][token[0]].lower().strip()
    else:
        resolved_token = token.lower().strip()

    # If the token is one or more number words, convert it to a number.
    is_worded_number = all(word_num_checker_dict.get(subtoken) is not None for subtoken in resolved_token.split(' ')) and resolved_token != 'and' and resolved_token != 'point'
    if (is_worded_number):
        resolved_token = w2n.word_to_num(resolved_token)
        if (type(resolved_token) is float and resolved_token.is_integer()):
            resolved_token = str(int(resolved_token))
        else:
            resolved_token = str(resolved_token)

    return resolved_token

def update_vocab(qar):
    sentences = qar['question'], *qar['answer_choices'], *qar['rationale_choices']
    metadata = open_image_metadata_file(qar)

    # Iterate through all tokens in the QAR, replacing any object references with their classname where appropriate.
    # for token in itertools.chain(*sentences):
    for sentence in sentences:
        for i, token in enumerate(sentence):
            # Resolve the token if the token is a reference to an object.
            resolved_token = preprocess_token(token, qar, metadata)

            sentence[i] = resolved_token

            vocab[resolved_token] += 1
            corpus.append(resolved_token)

def extract_folds_from_file_set(file_set, params):
    load_answers = params['load_answers']
    with open(os.path.join(annotations_dir, file_set)) as f:
        json_qars = list(f)
    
    qar_count = len(json_qars)
    qar_count_digits = len('%s' % qar_count)
    print(f'Loading {qar_count} Question-Answer-Rationale objects')

    for i, json_qar in enumerate(json_qars):
        if i % 100 == 0:
            print(f'({i + 1:0{qar_count_digits}}/{qar_count}) Processing QAR of file {file_set}')

        qar = json.loads(json_qar)

        split, qar_split_id = qar['annot_id'].split('-')
        qar_split_id = int(qar_split_id)
        match_fold = qar['match_fold']

        # Assign ids to respective dictionaries.

        if (match_fold not in split_folds):
            split_folds[match_fold] = []
        if (file_set not in file_splits):
            file_splits[file_set] = []
        if (match_fold not in file_splits[file_set]):
            file_splits[file_set].append(match_fold)

        qid2que[qar_split_id] = qar['question']
        split_folds[match_fold].append(qar)

        # Update the vocabulary with any new values.
        update_vocab(qar)

        # Update the answers file if enabled.
        if (load_answers):
            if (split not in split_answers):
                split_answers[split] = [ '<unk>' ]
            answer_i = [i for i in qar['answer_match_iter'] if qar['answer_match_iter'][i] == 0][0]
            split_answers[split].append(qar['answer_choices'][answer_i])

def build_imdb(fold_name, with_answers = True):
    imdb = []

    print(f'Constructing imdb for fold {fold_name}.')
    qar_count = len(split_folds[fold_name])
    qar_count_digits = len('%s' % qar_count)
    for i, qar in enumerate(split_folds[fold_name]):
        if i % 1000 == 0:
            print(f'({i + 1:0{qar_count_digits}}/{qar_count}) Processing imdb entry.')

        image_name = os.path.splitext(os.path.basename(qar['img_fn']))[0]
        image_path = os.path.realpath(os.path.join(images_dir, qar['img_fn']))
        image_id = int(qar['img_id'].split('-')[1])
        question_id = int(qar['annot_id'].split('-')[1])
        feature_path = os.path.realpath(os.path.join(resnet_feature_dir, os.path.splitext(qar['img_fn'])[0] + '.npy'))
        question_tokens = qar['question']
        question_str = ' '.join(question_tokens)

        imdb_entry = {
            'image_name': image_name,
            'image_path': image_path,
            'image_id': image_id,
            'question_id': question_id,
            'feature_path': feature_path,
            'question_str': question_str,
            'question_tokens': question_tokens,
        }

        imdb_entry['all_answers'] = qar['answer_choices']
        imdb_entry['all_rationales'] = qar['rationale_choices']

        if with_answers:
            imdb_entry['valid_answers'] = qar['answer_match_iter']
            imdb_entry['valid_rationales'] = qar['rationale_match_iter']

        imdb.append(imdb_entry)

    print(f'Processing completed for fold {fold_name}')
    return imdb


print(f'Loading {len(file_sets)} file sets.')
for file_set, params in file_sets.items():
    print(f'Processing {file_set}.')
    extract_folds_from_file_set(file_set, params)
    print(f'Completed {file_set}.')
    print(f'Completed {len(file_splits[file_set])} folds.')
    print(f'Loaded {sum(len(split_folds[fold]) for fold in file_splits[file_set])} QAR sets across all folds.')

print('File-loading completed!')

os.makedirs(imdb_out_dir, exist_ok=True)

# Write the current vocabulary to the file.
print(f'Saving vocabulary file to {vocabulary_file}')
with open(vocabulary_file, 'w') as f:
    f.writelines(f'{token[0]} {token[1]}\n' for token in sorted(vocab.items()))
with open(vocabulary_file, 'w') as f:
    f.writelines(f'{token[0]} {token[1]}\n' for token in sorted(vocab.items()))
# Write the corpus to the file.
print(f'Saving corpus file to {corpus_file}')
with open(corpus_file, 'w') as f:
    f.writelines(token + ' ' for token in corpus)

# Write the answer vocabs
for split, answers in split_answers.items():
    if (len(answers) != 0):
        answer_path = answers_file % split
        print(f'Saving {split} answers to {answer_path}')

        with open(answer_path, 'w') as f:
            f.writelines(f"{' '.join(token)}\n" for token in answers)

print('Constructing and saving imdbs')
for file_set, params in file_sets.items():
    imdb = []

    for fold_names in file_splits[file_set]:
        imdb += build_imdb(fold_names, with_answers=params['load_answers'])

    longest_token_sequences[file_set] = {
        'question': max((entry['question_id'], len(entry['question_tokens'])) for entry in imdb),
        'answer': max((entry['question_id'], len(answer)) for entry in imdb for answer in entry['all_answers']),
        'rationale': max((entry['question_id'], len(rationale)) for entry in imdb for rationale in entry['all_rationales']),
    }

    imdb_filename = f'imdb_{os.path.splitext(file_set)[0]}.npy'
    
    print(f'Saving imdb {imdb_filename}')
    np.save(os.path.join(imdb_out_dir, imdb_filename), np.array(imdb))

for key, entry in longest_token_sequences.items():
    print(f'Longest token sequences for {key}')
    print(f"  * Question: {entry['question'][1]} tokens from question '{entry['question'][0]}'")
    print(f"  * Answer: {entry['answer'][1]} tokens from question '{entry['answer'][0]}'")
    print(f"  * Rationale: {entry['rationale'][1]} tokens from question '{entry['rationale'][0]}'")
