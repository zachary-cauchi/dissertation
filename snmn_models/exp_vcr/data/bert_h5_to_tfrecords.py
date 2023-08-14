import os
import tqdm
import argparse
import tensorflow as tf

import bert_handler
import tfrecords_helpers

parser = argparse.ArgumentParser(prog='bert_h5_to_tfrecords.py', description='Convert the source BERT files from H5 datasets to tfrecords.')
parser.add_argument('--imdb_file_dir', type=str, default='tfrecords_imdb_r152_7x7')
parser.add_argument('--bert_src_dir', type=str, default='./bert_embeddings/')

args = parser.parse_args()
bert_dst_base_dir = './tfrecords_bert_embeddings'

imdb_bert_sets = [
    ('train', 'imdb_train.tfrecords', 'bert_da_answer_train.h5', 'bert_da_rationale_train.h5', True),
    ('val', 'imdb_val.tfrecords', 'bert_da_answer_val.h5', 'bert_da_rationale_val.h5', True),
    ('test', 'imdb_test.tfrecords', 'bert_da_answer_test.h5', 'bert_da_rationale_test.h5', False)
]

options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP, compression_level=9)

def get_imdb_ids(imdb_file, load_correct_answer = True):
    imdb_dataset = tf.data.TFRecordDataset(imdb_file)
    if load_correct_answer:
        imdb_dataset = imdb_dataset.map(lambda x: tfrecords_helpers.parse_example_to_imdb_with_correct_answer(x)['question_id'], num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        imdb_dataset = imdb_dataset.map(lambda x: tfrecords_helpers.parse_example_to_imdb_no_correct_answer(x)['question_id'], num_parallel_calls=tf.data.experimental.AUTOTUNE)
    imdb_dataset = imdb_dataset.batch(4096)

    final_ids_list = []

    with tf.variable_scope('get_ids_scope', reuse=False):
        with tf.compat.v1.Session() as sess:
            iterator = imdb_dataset.make_one_shot_iterator()
            next_element = iterator.get_next()
            try:
                while True:
                    final_ids_list.extend(sess.run(next_element))
            except tf.errors.OutOfRangeError:
                pass
            del iterator, next_element, imdb_dataset

    final_ids_list.sort()
    return final_ids_list

def save_to_tfrecords(dst_file, ctx_ans, ans, ctx_rat, rat):
    with tf.io.TFRecordWriter(dst_file) as writer:
        serialised_entry = tfrecords_helpers.serialize_both_bert_embeds_to_example(ctx_ans, ans, ctx_rat, rat)
        # parsed_entry = tfrecords_helpers.parse_example_to_both_bert_embeds(serialised_entry)
        # with tf.compat.v1.Session() as sess:
        #     entry = sess.run(parsed_entry)
        writer.write(serialised_entry)

for set_name, imdb_file, ans_h5_file, rat_h5_file, load_correct_answer in imdb_bert_sets:
    final_ids_list = get_imdb_ids(os.path.join(args.imdb_file_dir, imdb_file), load_correct_answer)
    handler = bert_handler.BertHandler(os.path.join(args.bert_src_dir, ans_h5_file), os.path.join(args.bert_src_dir, rat_h5_file), load_correct_answer)

    bert_dst_set_dir = os.path.join(bert_dst_base_dir, set_name)
    os.makedirs(bert_dst_set_dir, exist_ok=True)

    for id in tqdm.tqdm(final_ids_list, desc=f'Parsing {set_name}'):
        embeds = handler.get_embeddings_by_id(str(id))
        ctx_ans, ans = embeds['ans']
        ctx_rat, rat = embeds['rat']

        save_to_tfrecords(os.path.join(bert_dst_set_dir, str(id) + '.tfrecords'), ctx_ans, ans, ctx_rat, rat)
