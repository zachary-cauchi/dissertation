from functools import reduce
import os
from typing import Callable
import numpy as np
import tensorflow as tf
import json

from exp_vcr.data.bert_handler import BertHandler
import exp_vcr.data.tfrecords_helpers as tfrecords_helpers
from util import text_processing

class DataReader:
    def __init__(self, imdb_file, **data_params):
        self.print_fn = self.get_per_device_print_fn(tf.no_op('get_device_name'))
        self.imdb_file = imdb_file

        self.print_fn('Initialising imdb dataset')
        self.init_imdb_dataset()

        self.print_fn('Getting imdb count')
        self.get_imdb_count()

        self.data_params = data_params
        self.vcr_task_type = data_params['vcr_task_type']

        with tf.compat.v1.Session() as sess:
            iterator = self.imdb_dataset.make_one_shot_iterator()
            sample_elm = iterator.get_next()
            try:
                sample_record = sess.run(tfrecords_helpers.parse_example_to_imdb_with_correct_answer(sample_elm))
            except:
                sample_record = sess.run(tfrecords_helpers.parse_example_to_imdb_no_correct_answer(sample_elm))
            sess.close()

        if 'external_true_answers_file' in data_params and len(data_params['external_true_answers_file']) > 0:
            self.print_fn(f'Loading external truth answers from {data_params["external_true_answers_file"]}')
            with open(data_params['external_true_answers_file']) as f:
                self.use_external_true_answers = True
                raw_true_answers_data = f.read()
                true_answers_data = json.loads(raw_true_answers_data)
                true_answers_data = sorted(true_answers_data, key = lambda x: x['question_id'])
                self.true_answers_map = [ entry['answer'] for entry in true_answers_data ]

                assert len(self.true_answers_map) == len(true_answers_data), 'The true answer data does not match properly. Problem loading.'
                del true_answers_data
        else:
            self.use_external_true_answers = False

        self.vocab_dict = text_processing.VocabDict(
            data_params['vocab_question_file'], first_token_only=True)
        self.T_q_encoder = data_params['T_q_encoder']
        self.T_a_encoder = data_params['T_a_encoder']
        self.T_r_encoder = data_params['T_r_encoder']

        # peek one example to see whether answer and gt_layout are in the data
        self.load_correct_answer = (
            'valid_answers' in sample_record or self.use_external_true_answers)
        self.load_correct_rationale = (
            'valid_rationales' in sample_record)
        self.load_gt_layout = (
            ('load_gt_layout' in data_params and data_params['load_gt_layout'])
            and ('gt_layout_qa_tokens' in sample_record and
                 sample_record['gt_layout_qa_tokens'] is not None))
        self.load_rationale = data_params['vcr_task_type'] == 'QA_2_R' or data_params['vcr_task_type'] == 'Q_2_AR'

        if data_params['load_bert_embeddings'] == True:
            self.print_fn('Loading BERT embeddings.')
            self.load_bert = True
            self.bert_path = data_params['bert_embeddings_path']
            self.bert_path = self.bert_path if self.bert_path.endswith('/') else self.bert_path + '/'

            # Load a sample to determine dimensionality of each BERT vector.
            with tf.compat.v1.Session() as sess:
                self.print_fn('Loading BERT sample.')
                bert_sample_dataset = tf.data.TFRecordDataset(self.bert_path + '0.tfrecords')
                iter = bert_sample_dataset.make_one_shot_iterator()
                next_elem = iter.get_next()
                bert_sample = sess.run(tfrecords_helpers.parse_example_to_both_bert_embeds(next_elem))
                self.bert_dim = bert_sample[0]['ctx'][0].shape[1]
                del bert_sample_dataset, iter, next_elem
        else:
            self.load_bert = False

        self.num_answers = len(sample_record['all_answers'])
        self.num_rationales = len(sample_record['all_rationales'])

        if self.data_params['vcr_task_type'] == 'Q_2_A':
            self.correct_label_batch_name = 'valid_answer_onehot'
        elif self.data_params['vcr_task_type'] == 'QA_2_R':
            self.correct_label_batch_name = 'valid_rationale_onehot'
        elif self.data_params['vcr_task_type'] == 'Q_2_AR':
            self.correct_label_batch_name = 'valid_answer_and_rationale_onehot'

        if data_params['vcr_task_type'] == 'Q_2_A':
            self.num_combinations = self.num_answers
        elif data_params['vcr_task_type'] == 'QA_2_R':
            if self.load_correct_answer:
                self.num_combinations = self.num_rationales
            else:
                self.num_combinations = self.num_answers * self.num_rationales
        elif data_params['vcr_task_type'] == 'Q_2_AR':
            self.num_combinations = self.num_answers * self.num_rationales

        # Precompute the sequence of answer and rationale access ahead of when they're needed.
        i_ans_divisor = self.num_combinations // self.num_answers
        i_rat_mod = max(self.num_combinations // self.num_rationales, self.num_rationales)
        self.i_ans_range = [0] * self.num_combinations
        self.i_rat_range = [0] * self.num_combinations

        for i in range(self.num_combinations):
            self.i_ans_range[i] = i // i_ans_divisor
            self.i_rat_range[i] = i % i_rat_mod

        self.grouped_batch_size = data_params['batch_size']
        self.actual_batch_size = data_params['batch_size'] * self.num_combinations

        if not self.load_correct_answer:
            self.print_fn('imdb does not contain correct answers')
        if not self.load_correct_rationale:
            self.print_fn('imdb does not contain correct rationales')

        if not self.load_rationale:
            self.print_fn('Model is running in neither QA_2_R nor Q_2_AR')

        self.T_decoder = data_params['T_decoder']

        self.layout_dict = text_processing.VocabDict(
            data_params['vocab_layout_file'])
        if self.load_gt_layout:
            # Prune multiple filter modules by default
            self.prune_filter_module = (
                data_params['prune_filter_module']
                if 'prune_filter_module' in data_params else True)
        else:
            self.print_fn('imdb does not contain ground-truth layout')
        # Whether to load soft scores (targets for sigmoid regression)
        self.load_soft_score = ('load_soft_score' in data_params and
                                data_params['load_soft_score'])

        # load one feature map to peek its size
        sample_feature_path = sample_record['feature_path'].decode('utf-8')
        self.feature_parent_dir = os.path.dirname(os.path.dirname(sample_feature_path))

        self.print_fn('Loading sample image features to peek size.')
        self.feature_file_type = sample_feature_path.split('.')[-1]
        if self.feature_file_type == 'npy':
            self.feature_file_size = os.path.getsize(sample_feature_path)
            feats = np.load(sample_feature_path)
        elif self.feature_file_type == 'tfrecords':
            with tf.compat.v1.Session() as sess:
                sample_feature_reader = tf.data.TFRecordDataset(sample_feature_path)
                iterator = sample_feature_reader.make_one_shot_iterator()
                sample_feature_next_element = iterator.get_next()
                feats = sess.run(tfrecords_helpers.parse_resnet_example_to_nparray(sample_feature_next_element))
                del sample_feature_next_element, iterator, sample_feature_reader
                sess.close()
        else:
            raise ValueError(f'Feature file type not supported ({self.feature_file_type})')

        _, self.feat_H, self.feat_W, self.feat_D = feats.shape

    def get_imdb_count(self):
        if not hasattr(self, 'imdb_dataset'):
            self.init_imdb_dataset()

        with tf.variable_scope('imdb_counter'):
            with tf.compat.v1.Session() as sess:
                iterator = self.imdb_dataset.batch(4096).make_one_shot_iterator()
                next_element = iterator.get_next()
                counter = tf.Variable(0, dtype=tf.int32)
                increment_op = tf.assign_add(counter, tf.shape(next_element)[0])
                sess.run(tf.initialize_variables([ counter ]))
                try:
                    while True:
                        sess.run([increment_op, next_element])
                except tf.errors.OutOfRangeError:
                    pass
                self.imdb_count = sess.run(counter)
                del iterator, next_element, counter, increment_op

    def init_imdb_dataset(self):
        if self.imdb_file.endswith('.npy'):
            self.print_fn(f'Loading imdb npy dataset from {self.imdb_file}')
            imdb = np.load(self.imdb_file, allow_pickle=True)
            self.imdb_dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(imdb))
        elif self.imdb_file.endswith('.tfrecords'):
            self.print_fn(f'Initialising imdb TFRecords dataset from {self.imdb_file}')
            self.imdb_dataset: tf.data.Dataset = tf.compat.v1.data.TFRecordDataset(self.imdb_file)
        else:
            raise TypeError('unknown imdb format.')

    def init_dataset(self):
        self.init_imdb_dataset()

        # Deserialise the tensor data.
        final_dataset = self.imdb_dataset
        final_dataset: tf.data.Dataset = final_dataset.map(self.parse_raw_tensors, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Cache the dataset and shuffle the records.
        final_dataset = final_dataset.cache().shuffle(buffer_size=self.imdb_count, reshuffle_each_iteration=True)
        # Load BERT embeddings if enabled.
        if self.load_bert:
            final_dataset = final_dataset.interleave(self.load_bert_embeddings, cycle_length=8, block_length=1, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # Load the image features using the imdb['feature_path']
        final_dataset = final_dataset.interleave(self.load_image_features, cycle_length=8, block_length=1, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # First batch these elements before splitting.
        # final_dataset = final_dataset.batch(self.grouped_batch_size, drop_remainder=True).unbatch()
        # Split each imdb task into individual vcr tasks.
        final_dataset = final_dataset.flat_map(self.split_vcr_tasks)
        # Batch those tasks.
        final_dataset = final_dataset.batch(self.actual_batch_size, drop_remainder=False)
        # Perform final transpositions from nchw to nhwc.
        final_dataset = final_dataset.map(self.to_time_major, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.load_correct_answer:
            final_dataset = final_dataset.map(lambda x: (x, x[self.correct_label_batch_name]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        final_dataset = final_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return final_dataset

    def load_bert_embeddings(self, imdb):
        bert_dataset = tf.data.TFRecordDataset(tf.strings.join([self.bert_path, tf.as_string(imdb['question_id']), '.tfrecords'], name='bert_path'))

        def add_bert_to_imdb(feat):
            bert = tfrecords_helpers.parse_example_to_both_bert_embeds(feat)

            return_data = {
                **imdb,
            }

            if self.vcr_task_type == 'Q_2_A':
                # Extract the embeddings from the answer set only.
                bert_question_embedding = bert[0]['ctx']
                bert_answer_embedding = bert[0]['ans']
                bert_rationale_embedding = None
            elif self.vcr_task_type == 'QA_2_R' or self.vcr_task_type == 'Q_2_AR':
                # Extract the embeddings from the rationale set.
                bert_qa_embedding = bert[1]['ctx']
                question_length = imdb['question_length']

                # The Question and Answer embeddings are joined and need to be separated.
                bert_question_embedding = [ tf.slice(embedding, [0, 0], [question_length, tf.shape(embedding)[1]]) for embedding in bert_qa_embedding]
                bert_answer_embedding = [ tf.slice(embedding, [question_length, 0], [tf.shape(embedding)[0] - question_length, tf.shape(embedding)[1]]) for embedding in bert_qa_embedding ]
                bert_rationale_embedding = bert[1]['rat']
            else:
                raise ValueError(f'Unsupported task type for BERT embeddings {self.vcr_task_type}')

            # Apply padding to the embeddings and save them to the imdb.

            return_data.update({
                'bert_question_embedding': self.pad_or_trim_2d_list(bert_question_embedding, max_length=self.T_q_encoder, padding=0.),
                'bert_answer_embedding': self.pad_or_trim_2d_list(bert_answer_embedding, max_length=self.T_a_encoder, padding=0.),
            })

            if bert_rationale_embedding is not None:
                return_data['bert_rationale_embedding'] = self.pad_or_trim_2d_list(bert_rationale_embedding, max_length=self.T_r_encoder, padding=0.)

            return return_data

        return bert_dataset.map(add_bert_to_imdb)

    def load_image_features(self, imdb):
        image_feature_dataset = tf.data.TFRecordDataset(imdb['feature_path'])

        def add_resnet_to_imdb(feat):
            return { **imdb, 'image_feat': tf.ensure_shape(tfrecords_helpers.parse_resnet_example_to_nparray(feat)[0], (self.feat_H, self.feat_W, self.feat_D)) }

        return image_feature_dataset.map(add_resnet_to_imdb)

    def split_vcr_tasks(self, sample):
        def get_vcr_task(i, i_ans, i_rat):
            new_sample = {
                'image_name': sample['image_name'],
                'image_path': sample['image_path'],
                'image_feat': sample['image_feat'],
                'image_id': sample['image_id'],
                'feature_path': sample['feature_path'],
                'question_id': sample['question_id'],
                'question_str': sample['question_str'],
                'question_tokens': sample['question_tokens'],
                'question_sequence': sample['question_sequence'],
                'question_length': sample['question_length'],
                'all_answers': sample['all_answers'][i_ans],
                'all_answers_sequences': sample['all_answers_sequences'][i_ans],
                'all_answers_length': sample['all_answers_length'][i_ans],
                'all_rationales': sample['all_rationales'][i_rat],
                'all_rationales_sequences': sample['all_rationales_sequences'][i_rat],
                'all_rationales_length': sample['all_rationales_length'][i_rat]
            }

            if self.load_correct_answer:
                new_sample.update({
                    'valid_answers': sample['valid_answers'][i_ans],
                    'valid_answer_index': sample['valid_answer_index'],
                    'valid_answer_onehot': sample['valid_answer_onehot'][i]
                })

            if self.load_correct_rationale:
                new_sample.update({
                    'valid_rationales': sample['valid_rationales'][i_rat],
                    'valid_rationale_index': sample['valid_rationale_index'],
                    'valid_rationale_onehot': sample['valid_rationale_onehot'][i]
                })

            if 'valid_answer_and_rationale_onehot' in sample:
                new_sample['valid_answer_and_rationale_onehot'] = sample['valid_answer_and_rationale_onehot'][i]

            if self.load_bert:
                if self.vcr_task_type == 'Q_2_A':
                    new_sample.update({
                        # Don't know why bert_question_embedding has the extra dimension, but it must be accounted for.
                        'bert_question_embedding': sample['bert_question_embedding'][i_ans],
                        'bert_answer_embedding': sample['bert_answer_embedding'][i_ans]
                    })
                elif self.vcr_task_type == 'QA_2_R' or self.vcr_task_type == 'Q_2_AR':
                    if self.load_correct_answer:
                        new_sample.update({
                            'bert_question_embedding': sample['bert_question_embedding'][i_rat],
                            'bert_answer_embedding': sample['bert_answer_embedding'][i_ans],
                            'bert_rationale_embedding': sample['bert_rationale_embedding'][i_rat]
                        })
                    else:
                        new_sample.update({
                            'bert_question_embedding': sample['bert_question_embedding'][i],
                            'bert_answer_embedding': sample['bert_answer_embedding'][i],
                            'bert_rationale_embedding': sample['bert_rationale_embedding'][i]
                        })
                else:
                    raise ValueError(f'Unsupported task type for BERT embeddings{self.vcr_task_type}')

            return new_sample

        datasets = []
        app_sample = datasets.append

        for i in range(self.num_combinations):
            if self.vcr_task_type == 'Q_2_A':
                    i_ans = self.i_ans_range[i]
                    i_rat = 0
            else:
                i_ans = sample['valid_answer_index'] if 'valid_answer_index' in sample else self.i_ans_range[i]
                i_rat = self.i_rat_range[i]

            app_sample(tf.data.Dataset.from_tensors(get_vcr_task(i, i_ans, i_rat)))

        return reduce(lambda ds1, ds2: ds1.concatenate(ds2), datasets)

    def parse_raw_tensors(self, imdb_sample):
        if self.load_correct_answer:
            imdb = tfrecords_helpers.parse_example_to_imdb_with_correct_answer(imdb_sample)
        else:
            imdb = tfrecords_helpers.parse_example_to_imdb_no_correct_answer(imdb_sample)

        # Pad or trim each sequence to the fixed encoder length.
        imdb['question_tokens'] = self.pad_or_trim(imdb['question_tokens'], self.T_q_encoder)
        imdb['question_sequence'] = self.pad_or_trim(imdb['question_sequence'], self.T_q_encoder, padding=0)
        imdb['all_answers'] = self.pad_or_trim_1d_list(imdb['all_answers'], self.T_a_encoder)
        imdb['all_answers_sequences'] = self.pad_or_trim_1d_list(imdb['all_answers_sequences'], self.T_a_encoder, padding=0)
        imdb['all_rationales'] = self.pad_or_trim_1d_list(imdb['all_rationales'], self.T_r_encoder)
        imdb['all_rationales_sequences'] = self.pad_or_trim_1d_list(imdb['all_rationales_sequences'], self.T_r_encoder, padding=0)

        # Cap the sequence lengths to their target length
        imdb['question_length'] = tf.math.minimum(imdb['question_length'], self.T_q_encoder, name='cap_question_length')
        imdb['all_answers_length'] = tf.math.minimum(imdb['all_answers_length'], self.T_a_encoder, name='cap_answer_length')
        imdb['all_rationales_length'] = tf.math.minimum(imdb['all_rationales_length'], self.T_r_encoder, name='cap_rationale_length')

        # If we are using externally sourced true answers, load them in now.
        if self.use_external_true_answers:
            imdb['valid_answer_index'] = tf.gather(self.true_answers_map, imdb['question_id'])

        if self.load_correct_answer:
            imdb['valid_answer_onehot'] = tf.one_hot(imdb['valid_answer_index'], self.num_answers, 1., 0.)
        if self.load_correct_rationale:
            imdb['valid_rationale_onehot'] = tf.one_hot(imdb['valid_rationale_index'], self.num_rationales, 1., 0.)
        if self.load_correct_answer and self.load_correct_rationale and self.vcr_task_type == 'Q_2_AR':
            imdb['valid_answer_onehot'] = tf.tile(input=imdb['valid_answer_onehot'], multiples=[ self.num_rationales ])
            imdb['valid_rationale_onehot'] = tf.repeat(imdb['valid_rationale_onehot'], repeats=self.num_answers)
            imdb['valid_answer_and_rationale_onehot'] = tf.math.logical_and(tf.cast(imdb['valid_answer_onehot'], tf.bool), tf.cast(imdb['valid_rationale_onehot'], tf.bool))
            imdb['valid_answer_and_rationale_onehot'] = tf.cast(imdb['valid_answer_and_rationale_onehot'], tf.float32)

        return imdb

    def pad_or_trim(self, tensor, length, padding = ''):
        # Determine the current length of the tensor
        current_length = tf.shape(tensor)[0]

        # Calculate the number of elements to pad
        num_pad = tf.maximum(0, length - current_length)

        # Pad the tensor
        tensor = tf.pad(tensor, [[0, num_pad]], constant_values=padding)

        # Trim the tensor to the maximum length
        tensor = tf.slice(tensor, [0], [length])

        return tensor

    def pad_or_trim_1d_list(self, tensor_list, max_length, padding=''):
        return [ self.pad_or_trim(tensor, max_length, padding=padding) for tensor in tensor_list ]

    def pad_or_trim_2d(self, tensor, max_length, padding=''):
        current_length = tf.shape(tensor)[0]

        # Condition: if current_length is less than max_length
        def pad():
            padding_needed = tf.subtract(max_length, current_length)  # use tf.subtract
            paddings = tf.concat([tf.constant([0]), tf.reshape(padding_needed, [1])], axis=0)
            paddings_matrix = tf.stack([paddings, tf.constant([0, 0])], axis=0)
            return tf.pad(tensor, paddings_matrix, constant_values=padding)

        # Condition: if current_length is greater than max_length
        def slice_():
            return tf.slice(tensor, [0, 0], [max_length, -1])

        # Use tf.cond to decide between padding or slicing based on tensor's current length
        return tf.cond(current_length < max_length, pad, slice_)

    def pad_or_trim_2d_list(self, tensor_list, max_length, padding=''):
        return [ self.pad_or_trim_2d(tensor, max_length, padding=padding) for tensor in tensor_list ]

    def to_time_major(self, element):
        element['question_sequence'] = tf.transpose(element['question_sequence'])
        element['all_answers_sequences'] = tf.transpose(element['all_answers_sequences'])
        if self.load_rationale:
            element['all_rationales_sequences'] = tf.transpose(element['all_rationales_sequences'])
        if self.load_bert:
            element['bert_question_embedding'] = tf.ensure_shape(tf.transpose(element['bert_question_embedding'], [1, 0, 2]), (self.T_q_encoder, element['bert_question_embedding'].shape[0], self.bert_dim))
            element['bert_answer_embedding'] = tf.ensure_shape(tf.transpose(element['bert_answer_embedding'], [1, 0, 2]), (self.T_a_encoder, element['bert_answer_embedding'].shape[0], self.bert_dim))
            if self.load_rationale:
                element['bert_rationale_embedding'] = tf.ensure_shape(tf.transpose(element['bert_rationale_embedding'], [1, 0, 2]), (self.T_r_encoder, element['bert_rationale_embedding'].shape[0], self.bert_dim))
        return element

    def __del__(self):
        if hasattr(self, 'bert_handler'):
            del bert_handler

    def get_per_device_print_fn(self, tensor) -> Callable[[str], None]:
        device = tensor.device
        return lambda msg: print(f'{device + ": " if device is not None else ""}DataReader: {msg}')
