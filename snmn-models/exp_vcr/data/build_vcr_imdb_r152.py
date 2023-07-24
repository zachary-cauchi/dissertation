import numpy as np
import json
import os
import argparse

import sys; sys.path.append('../../')  # NOQA
from collections import Counter
from word2number import w2n

parser = argparse.ArgumentParser(prog='build_vcr_imdb_r152.py', description='Build the VCR imdb files.')
parser.add_argument('--res', type=str, default='8', required=False,
                    help='Size of the resnet features (eg. \'--res 8\' will look for 8x8 resnets)')
args = parser.parse_args()

annotations_dir = '../vcr_dataset/Annotations'
images_dir = '../vcr_dataset/vcr1images'
corpus_file = './corpus_vcr.txt'
corpus_bert_file = './corpus_bert_vcr.txt'
vocabulary_file = './vocabulary_vcr.txt'
answers_file = './answers_%s_vcr.txt'
rationales_file = './rationales_%s_vcr.txt'
imdb_out_dir = f'./imdb_r152_{args.res}x{args.res}'
resnet_feature_dir = f'./resnet152_c5_{args.res}x{args.res}'
file_sets = {
    'train.jsonl': { 'load_answers': True, 'load_rationales': True },
    'val.jsonl': { 'load_answers': True, 'load_rationales': True },
    'test.jsonl': { 'load_answers': False, 'load_rationales': False }
}

token_delimeter = ' '
# For BERT use ' ||| ', for the Stanford Parser use ' '.
sentence_delimeter = ' ||| '

file_splits = {}
split_folds = {}
img2qid = {}
qid2que = {}
corpus = []
vocab = Counter()
split_answers = {}
split_rationales = {}

bert_corpus = []

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
    is_worded_number = all(word_num_checker_dict.get(subtoken) is not None for subtoken in resolved_token.split(token_delimeter)) and resolved_token != 'and' and resolved_token != 'point'
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
            # TODO: Add correctness check to prevent duplicate/wrong sentences.
            corpus.append(resolved_token)

def extract_folds_from_file_set(file_set, params):
    load_answers = params['load_answers']
    load_rationales = params['load_rationales']

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

        corpus_entry = token_delimeter.join(qar['question'])

        # Update the answers file if enabled.
        if (load_answers):
            if (split not in split_answers):
                split_answers[split] = [ '<unk>' ]
            answer_i = [i for i in qar['answer_match_iter'] if qar['answer_match_iter'][i] == 0][0]
            split_answers[split].append(qar['answer_choices'][answer_i])
            corpus_entry += sentence_delimeter + token_delimeter.join(qar['answer_choices'][answer_i])
        
        # Update the rationales file if enabled.
        if (load_rationales):
            if (split not in split_rationales):
                split_rationales[split] = [ '<unk>' ]
            rationale_i = [i for i in qar['rationale_match_iter'] if qar['rationale_match_iter'][i] == 0][0]
            split_rationales[split].append(qar['rationale_choices'][rationale_i])
            corpus_entry += sentence_delimeter + token_delimeter.join(qar['rationale_choices'][rationale_i])
        
        bert_corpus.append(corpus_entry + '\n')

def build_imdb(fold_name, with_answers = True, with_rationales = True):
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
        question_str = token_delimeter.join(question_tokens)

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

# Write the corpuses to the files.

print(f'Saving corpus file to {corpus_file}')
with open(corpus_file, 'w') as f:
    f.writelines(token + token_delimeter for token in corpus)

print(f'Saving BERT corpus file to {corpus_bert_file}')
with open(corpus_bert_file, 'w') as f:
    f.writelines(entry for entry in bert_corpus)

# Write the answer vocabs
for split, answers in split_answers.items():
    if (len(answers) != 0):
        answer_path = answers_file % split
        print(f'Saving {split} answers to {answer_path}')

        with open(answer_path, 'w') as f:
            f.writelines(f"{token_delimeter.join(token)}\n" for token in answers)

# Write the rationale vocabs
for split, rationales in split_rationales.items():
    if (len(rationales) != 0):
        rationale_path = rationales_file % split
        print(f'Saving {split} rationales to {rationales_file}')

        with open(rationale_path, 'w') as f:
            f.writelines(f"{token_delimeter.join(token)}\n" for token in rationales)

print('Constructing and saving imdbs')
for file_set, params in file_sets.items():
    imdb = []

    for fold_names in file_splits[file_set]:
        imdb += build_imdb(fold_names, with_answers=params['load_answers'], with_rationales=params['load_rationales'])

    qid = -1
    q_longest_len = -1
    aid = -1
    a_longest_len = -1
    rid = -1
    r_longest_len = -1

    q_length_counter = Counter()
    a_length_counter = Counter()
    r_length_counter = Counter()

    for entry in imdb:
        q_len = len(entry['question_tokens'])
        q_length_counter[q_len] += 1
        if (q_len > q_longest_len):
            qid = entry['question_id']
        
        for answer in entry['all_answers']:
            a_len = len(answer)
            a_length_counter[a_len] += 1
            if (a_len > a_longest_len):
                aid = entry['question_id']
        
        for rationale in entry['all_rationales']:
            r_len = len(rationale)
            r_length_counter[r_len] += 1
            if (r_len > r_longest_len):
                rid = entry['question_id']

    longest_token_sequences[file_set] = {
        'question': (qid, q_length_counter),
        'answer': (aid, a_length_counter),
        'rationale': (rid, r_length_counter),
    }

    imdb_filename = f'imdb_{os.path.splitext(file_set)[0]}.npy'
    
    print(f'Saving imdb {imdb_filename}')
    np.save(os.path.join(imdb_out_dir, imdb_filename), np.array(imdb))

with open('imdb_stats.txt', 'w') as stats:
    for key, entry in longest_token_sequences.items():
        q_counter: Counter = entry['question'][1]
        a_counter: Counter = entry['answer'][1]
        r_counter: Counter = entry['rationale'][1]

        msg = [
            f'Longest token sequences for {key}\n'
            f"  * Question: {list(q_counter.items())[0][0]} tokens from question \'{entry['question'][0]}\'\n"
            f"  * Answer: {list(a_counter.items())[0][0]} tokens from answer \'{entry['answer'][0]}\'\n"
            f"  * Rationale: {list(r_counter.items())[0][0]} tokens from rationale \'{entry['rationale'][0]}\'\n"
            '\n\nPrinting longest token occurrences:\n'
        ]
        
        q_count = sum(q_counter.values())
        a_count = sum(a_counter.values())
        r_count = sum(r_counter.values())

        msg.append('Questions:\n')
        msg.extend(f'  * {len}: {len_count // 4} ({len_count/q_count:.2%})\n' for len, len_count in sorted(q_counter.items(), key = lambda x: x[0], reverse = True))
        msg.append('\n\n')
        msg.append('Answers:\n')
        msg.extend(f'  * {len}: {len_count // 4} ({len_count/a_count:.2%})\n' for len, len_count in sorted(a_counter.items(), key = lambda x: x[0], reverse = True))
        msg.append('\n\n')
        msg.append('Rationales:\n')
        msg.extend(f'  * {len}: {len_count // 4} ({len_count/r_count:.2%})\n' for len, len_count in sorted(r_counter.items(), key = lambda x: x[0], reverse = True))
        msg.append('\n\n')

        msg = ''.join(msg)

        print(msg)
        stats.write(msg)
