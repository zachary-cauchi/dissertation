import numpy as np
import json
import os
import tqdm
import argparse
import tensorflow as tf
import tfrecords_helpers

import sys; sys.path.append('../../')  # NOQA
from collections import Counter
from word2number import w2n

parser = argparse.ArgumentParser(prog='build_vcr_imdb_r152.py', description='Build the VCR imdb files.')
parser.add_argument('--file_type', type=str, choices=['tfrecords', 'npy'], default='npy')
parser.add_argument('--res', type=str, default='8', required=False,
                    help='Size of the resnet features (eg. \'--res 8\' will look for 8x8 resnets)')
args = parser.parse_args()

file_type = args.file_type
annotations_dir = '../vcr_dataset/Annotations'
images_dir = '../vcr_dataset/vcr1images'
corpus_file = './corpus_vcr.txt'
corpus_bert_file = './corpus_bert_vcr.txt'
vocabulary_file = './vocabulary_vcr.txt'
answers_file = './answers_%s_vcr.txt'
rationales_file = './rationales_%s_vcr.txt'
imdb_out_dir = f'./imdb_r152_{args.res}x{args.res}' if file_type == 'npy' else f'./tfrecords_imdb_r152_{args.res}x{args.res}'
resnet_feature_dir = f'./resnet152_c5_{args.res}x{args.res}' if file_type == 'npy' else f'./tfrecords_resnet152_c5_{args.res}x{args.res}'
resnet_feature_ext = '.npy' if file_type == 'npy' else '.tfrecords'

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
vocab['<unk>'] += 1
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

def preprocess_token(token, qar, metadata) -> 'list[str]':
    # Resolve the token if the token is a reference to an object.
    # Since a token can be a list of one or more references, process all of them first.
    
    if isinstance(token, list):
        resolved_tokens = [ metadata['names'][t].lower().strip() for t in token ]
    else:
        resolved_tokens = [ token.lower().strip() ]

    for i, rt in enumerate(resolved_tokens):
        # If the token is one or more number words, convert it to a number.
        is_worded_number = all(word_num_checker_dict.get(subtoken) is not None for subtoken in rt.split(token_delimeter)) and rt != 'and' and rt != 'point'
        if (is_worded_number):
            rt = w2n.word_to_num(rt)
            if (type(rt) is float and rt.is_integer()):
                rt = str(int(rt))
            else:
                rt = str(rt)
            resolved_tokens[i] = rt

    return resolved_tokens

def update_vocab(qar):
    sentences: list[list[str]] = qar['question'], *qar['answer_choices'], *qar['rationale_choices']
    metadata = open_image_metadata_file(qar)

    # Iterate through all tokens in the QAR, replacing any object references with their classname where appropriate.
    # for token in itertools.chain(*sentences):
    for sentence in sentences:
        for i, token in enumerate(sentence):
            # Resolve the token if the token is a reference to an object.
            resolved_tokens = preprocess_token(token, qar, metadata)

            for j, rt in enumerate(resolved_tokens):
                if j == 0:
                    sentence[i] = rt
                else:
                    sentence.insert(i + j, rt)

                vocab[rt] += 1
                # TODO: Add correctness check to prevent duplicate/wrong sentences.
                corpus.append(rt)

def extract_folds_from_file_set(file_set, params):
    load_answers = params['load_answers']
    load_rationales = params['load_rationales']

    with open(os.path.join(annotations_dir, file_set)) as f:
        json_qars = list(f)
    
    qar_count = len(json_qars)
    print(f'Loading {qar_count} Question-Answer-Rationale objects')

    for json_qar in tqdm.tqdm(json_qars, desc=f'Processing QAR of file {file_set}'):
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
            answer_i = qar['answer_label']
            split_answers[split].append(qar['answer_choices'][answer_i])
            corpus_entry += sentence_delimeter + token_delimeter.join(qar['answer_choices'][answer_i])
        
        # Update the rationales file if enabled.
        if (load_rationales):
            if (split not in split_rationales):
                split_rationales[split] = [ '<unk>' ]
            rationale_i = qar['rationale_label']
            split_rationales[split].append(qar['rationale_choices'][rationale_i])
            corpus_entry += sentence_delimeter + token_delimeter.join(qar['rationale_choices'][rationale_i])
        
        bert_corpus.append(corpus_entry + '\n')

def build_imdb(fold_name, with_answers = True, with_rationales = True):
    imdb = []

    print(f'Constructing imdb for fold {fold_name}.')

    for qar in tqdm.tqdm(split_folds[fold_name], desc='Processing imdb entry'):

        image_name = os.path.splitext(os.path.basename(qar['img_fn']))[0]
        image_path = os.path.realpath(os.path.join(images_dir, qar['img_fn']))
        image_id = int(qar['img_id'].split('-')[1])
        question_id = int(qar['annot_id'].split('-')[1])
        feature_path = os.path.realpath(os.path.join(resnet_feature_dir, os.path.splitext(qar['img_fn'])[0] + resnet_feature_ext))
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
            imdb_entry['valid_answer_index'] = qar['answer_label']
        if with_rationales:
            imdb_entry['valid_rationales'] = qar['rationale_match_iter']
            imdb_entry['valid_rationale_index'] = qar['rationale_label']

        imdb.append(imdb_entry)

    print(f'Processing completed for fold {fold_name}')
    return imdb

def write_vocab_file(save_base_path, split_data):
    # Write the answer vocabs
    for split, data in split_data.items():
        if (len(data) != 0):
            answer_path = save_base_path % split
            print(f'Saving {split} answers to {answer_path}')

            with open(answer_path, 'w') as f:
                f.writelines(f"{token_delimeter.join(token)}\n" for token in data)

def export_to_tfrecords(file_path, imdb):
    with tf.python_io.TFRecordWriter(file_path) as writer:
        for entry in tqdm.tqdm(imdb, desc='Serializing to TFRecords'):
            serialized_entry = tfrecords_helpers.serialize_imdb_to_example(entry)

            writer.write(serialized_entry)


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
write_vocab_file(answers_file, split_answers)

# Write the rationale vocabs
write_vocab_file(rationales_file, split_rationales)

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
            q_longest_len = q_len
        
        for answer in entry['all_answers']:
            a_len = len(answer)
            a_length_counter[a_len] += 1
            if (a_len > a_longest_len):
                aid = entry['question_id']
                a_longest_len = a_len
        
        for rationale in entry['all_rationales']:
            r_len = len(rationale)
            r_length_counter[r_len] += 1
            if (r_len > r_longest_len):
                rid = entry['question_id']
                r_longest_len = r_len

    longest_token_sequences[file_set] = {
        'question': (qid, q_longest_len, q_length_counter),
        'answer': (aid, a_longest_len, a_length_counter),
        'rationale': (rid, r_longest_len, r_length_counter),
    }

    imdb_filename = f'imdb_{os.path.splitext(file_set)[0]}'
    imdb_filepath = os.path.join(imdb_out_dir, imdb_filename) + ('.npy' if file_type == 'npy' else '.tfrecords')

    if os.path.isfile(imdb_filepath):
        os.remove(imdb_filepath)

    print(f'Saving imdb {imdb_filename}')

    if file_type == 'npy':
        np.save(os.path.join(imdb_out_dir, imdb_filename) + '.npy', np.array(imdb))
    else:
        export_to_tfrecords(os.path.join(imdb_out_dir, imdb_filename) + '.tfrecords', imdb)

with open('imdb_stats.txt', 'w') as stats:
    for key, entry in longest_token_sequences.items():
        q_counter: Counter = entry['question'][2]
        a_counter: Counter = entry['answer'][2]
        r_counter: Counter = entry['rationale'][2]

        longest_q_token = sorted(q_counter.items(), key = lambda x: x[0], reverse = True)

        msg = [
            f'Longest token sequences for {key}\n'
            f"  * Question: {entry['question'][1]} tokens from question \'{entry['question'][0]}\'\n"
            f"  * Answer: {entry['answer'][1]} tokens from answer \'{entry['answer'][0]}\'\n"
            f"  * Rationale: {entry['rationale'][1]} tokens from rationale \'{entry['rationale'][0]}\'\n"
            '\n\nPrinting longest token occurrences:\n'
        ]
        
        q_count = sum(q_counter.values())
        a_count = sum(a_counter.values())
        r_count = sum(r_counter.values())

        msg.append('Questions:\n')
        msg.extend(f'  * {len}: {len_count} ({len_count/q_count:.2%})\n' for len, len_count in sorted(q_counter.items(), key = lambda x: x[0], reverse = True))
        msg.append('\n\n')
        msg.append('Answers:\n')
        msg.extend(f'  * {len}: {len_count} ({len_count/a_count:.2%})\n' for len, len_count in sorted(a_counter.items(), key = lambda x: x[0], reverse = True))
        msg.append('\n\n')
        msg.append('Rationales:\n')
        msg.extend(f'  * {len}: {len_count} ({len_count/r_count:.2%})\n' for len, len_count in sorted(r_counter.items(), key = lambda x: x[0], reverse = True))
        msg.append('\n\n')

        msg = ''.join(msg)

        print(msg)
        stats.write(msg)
