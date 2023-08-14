from functools import reduce
import os
from typing import Callable
import numpy as np
import tensorflow as tf

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

        self.vocab_dict = text_processing.VocabDict(
            data_params['vocab_question_file'], first_token_only=True)
        self.T_q_encoder = data_params['T_q_encoder']
        self.T_a_encoder = data_params['T_a_encoder']
        self.T_r_encoder = data_params['T_r_encoder']

        # peek one example to see whether answer and gt_layout are in the data
        self.load_correct_answer = (
            'valid_answers' in sample_record)
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

            with tf.compat.v1.Session() as sess:
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
            self.correct_label_batch_name = 'valid_rationale_index'
        elif self.data_params['vcr_task_type'] == 'Q_2_AR':
            self.correct_label_batch_name = 'valid_answer_and_rationale_index'

        if data_params['vcr_task_type'] == 'Q_2_A':
            self.num_combinations = self.num_answers
        elif data_params['vcr_task_type'] == 'QA_2_R':
            self.num_combinations = self.num_rationales
        else:
            self.num_combinations = self.num_answers * self.num_rationales

        # Precompute the sequence of answer and rationale access ahead of when they're needed.
        i_ans_divisor = self.num_combinations // self.num_answers
        i_rat_mod = self.num_combinations // self.num_rationales
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

        # Cache and shuffle the imdb dataset.
        final_dataset = self.imdb_dataset.cache().shuffle(buffer_size=self.imdb_count, reshuffle_each_iteration=True)
        final_dataset: tf.data.Dataset = final_dataset.map(self.parse_raw_tensors, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # Load the image features using the imdb['feature_path']
        final_dataset = final_dataset.interleave(self.load_image_features, cycle_length=8, block_length=1, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.load_bert:
            final_dataset = final_dataset.interleave(self.load_bert_embeddings, cycle_length=8, block_length=1, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # First batch these elements before splitting.
        final_dataset = final_dataset.batch(self.grouped_batch_size, drop_remainder=True).unbatch()
        # Split each imdb task into individual vcr tasks.
        final_dataset = final_dataset.flat_map(self.new_split_vcr_tasks)
        # Batch those tasks.
        final_dataset = final_dataset.batch(self.actual_batch_size, drop_remainder=True)
        # Perform final transpositions from nchw to nhwc.
        final_dataset = final_dataset.map(self.to_time_major, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.load_correct_answer:
            final_dataset = final_dataset.map(lambda x: (x, x[self.correct_label_batch_name]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        final_dataset = final_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        return final_dataset

    def load_bert_embeddings(self, imdb):
        bert_dataset = tf.data.TFRecordDataset(tf.strings.join([self.bert_path, tf.as_string(imdb['question_id']), '.tfrecords'], name='bert_path'))

        def add_to_imdb(feat):
            bert = tfrecords_helpers.parse_example_to_both_bert_embeds(feat)

            if self.vcr_task_type == 'Q_2_A':
                bert_question_embedding = self.pad_or_trim_2d_list(bert[0]['ctx'], max_length=self.T_q_encoder, padding=0.),
                bert_answer_embedding = self.pad_or_trim_2d_list(bert[0]['ans'], max_length=self.T_a_encoder, padding=0.)
                # bert_question_embedding = tf.ensure_shape(bert_question_embedding, [self.num_combinations, self.T_q_encoder, self.bert_dim])
                # bert_answer_embedding = tf.ensure_shape(bert_answer_embedding, [self.num_combinations, self.T_a_encoder, self.bert_dim])
            else:
                raise ValueError(f'Unsupported task type for BERT embeddings{self.vcr_task_type}')

            return {
                **imdb,
                'bert_question_embedding': bert_question_embedding,
                'bert_answer_embedding': bert_answer_embedding
            }
        
        return bert_dataset.map(add_to_imdb)

    def load_image_features(self, imdb):
        image_feature_dataset = tf.data.TFRecordDataset(imdb['feature_path'])

        def add_to_imdb(feat):
            return { **imdb, 'image_feat': tf.ensure_shape(tfrecords_helpers.parse_resnet_example_to_nparray(feat)[0], (self.feat_H, self.feat_W, self.feat_D)) }

        return image_feature_dataset.map(add_to_imdb)

    def new_split_vcr_tasks(self, sample):
        def map_fn(i_ans, i_rat):
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
                    'valid_answer_onehot': sample['valid_answer_onehot'][i_ans]
                })

            if self.load_correct_rationale:
                new_sample.update({
                    'valid_rationales': sample['valid_rationales'][i_rat],
                    'valid_rationale_index': sample['valid_rationale_index'],
                    'valid_rationale_onehot': sample['valid_rationale_onehot'][i_rat]
                })

            if self.load_bert:
                if self.vcr_task_type == 'Q_2_A':
                    new_sample.update({
                        # Don't know why bert_question_embedding has the extra dimension, but it must be accounted for.
                        'bert_question_embedding': sample['bert_question_embedding'][0][i_ans],
                        'bert_answer_embedding': sample['bert_answer_embedding'][i_ans]
                    })
                else:
                    raise ValueError(f'Unsupported task type for BERT embeddings{self.vcr_task_type}')

            return new_sample

        datasets = []
        app_sample = datasets.append

        for i_ans, i_rat in zip(self.i_ans_range, self.i_rat_range):
            app_sample(tf.data.Dataset.from_tensors(sample).map(lambda s: map_fn(i_ans, i_rat)))

        return reduce(lambda ds1, ds2: ds1.concatenate(ds2), datasets)

    def parse_raw_tensors(self, imdb_sample):
        if self.load_correct_answer:
            imdb = tfrecords_helpers.parse_example_to_imdb_with_correct_answer(imdb_sample)
        else:
            imdb = tfrecords_helpers.parse_example_to_imdb_no_correct_answer(imdb_sample)

        imdb['question_tokens'] = self.pad_or_trim(imdb['question_tokens'], self.T_q_encoder)
        imdb['question_sequence'] = self.pad_or_trim(imdb['question_sequence'], self.T_q_encoder, padding=0)
        imdb['all_answers'] = self.pad_or_trim_1d_list(imdb['all_answers'], self.T_a_encoder)
        imdb['all_answers_sequences'] = self.pad_or_trim_1d_list(imdb['all_answers_sequences'], self.T_a_encoder, padding=0)
        imdb['all_rationales'] = self.pad_or_trim_1d_list(imdb['all_rationales'], self.T_r_encoder)
        imdb['all_rationales_sequences'] = self.pad_or_trim_1d_list(imdb['all_rationales_sequences'], self.T_r_encoder, padding=0)

        if self.load_correct_answer:
            imdb['valid_answer_onehot'] = tf.one_hot(imdb['valid_answer_index'], self.num_combinations, 1., 0.)
        if self.load_correct_rationale:
            imdb['valid_rationale_onehot'] = tf.one_hot(imdb['valid_rationale_index'], self.num_combinations, 1., 0.)

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
            element['bert_question_embedding'] = tf.transpose(element['bert_question_embedding'], [0, 2, 1])
            element['bert_answer_embedding'] = tf.transpose(element['bert_answer_embedding'], [0, 2, 1])
            if self.load_rationale:
                element['bert_rationale_embedding'] = tf.transpose(element['bert_rationale_embedding'], [0, 2, 1])
        return element

    # def batches(self):
    #     num_samples = len(self.imdb)
    #     for sample_id in np.random.permutation(num_samples):
    #         collection = self.load_one(sample_id)

    #         for i in range(self.num_combinations):
    #             record = {}
    #             record['image_feat_batch'] = collection['image_feat_batch'][i]
    #             record['question_seq_batch'] = collection['question_seq_batch'][:, i]
    #             record['question_length_batch'] = collection['question_length_batch'][i]
    #             record['answer_index'] = collection['answer_index'][i]
    #             record['all_answers_seq_batch'] = collection['all_answers_seq_batch'][:, i]
    #             record['all_answers_length_batch'] = collection['all_answers_length_batch'][i]
    #             record['image_path_list'] = collection['image_path_list'][i]
    #             record['qid_list'] = collection['qid_list'][i]
    #             if self.load_correct_answer:
    #                 record['answer_label_batch'] = collection['answer_label_batch'][i]
    #             if self.load_rationale:
    #                 record['all_rationales_seq_batch'] = collection['all_rationales_seq_batch'][:, i]
    #                 record['all_rationales_length_batch'] = collection['all_rationales_length_batch'][i]
    #                 if self.load_correct_answer:
    #                     record['rationale_label_batch'] = collection['rationale_label_batch'][i]
    #             if self.load_bert:
    #                 record['bert_question_embeddings_batch'] = collection['bert_question_embeddings_batch'][:, i]
    #                 record['bert_answer_embeddings_batch'] = collection['bert_answer_embeddings_batch'][:, i]
    #                 if self.load_rationale:
    #                     record['bert_rationale_embeddings_batch'] = collection['bert_rationale_embeddings_batch'][:, i]
    #             yield record

    # def load_one(self, sample_id):

    #     # Allocate the arrays and collections.

    #     question_seq_batch = np.zeros(
    #         (self.T_q_encoder, self.num_combinations), np.int32)
    #     all_answers_seq_batch = np.zeros((self.T_a_encoder, self.num_combinations), np.int32)
    #     all_answers_length_batch = np.zeros((self.num_combinations), np.int32)
    #     if self.load_rationale:
    #         all_rationales_seq_batch = np.zeros((self.T_r_encoder, self.num_combinations), np.int32)
    #         all_rationales_length_batch = np.zeros((self.num_combinations), np.int32)
    #     if self.load_bert:
    #         bert_question_embeddings_batch = np.zeros((self.T_q_encoder, self.num_combinations, self.bert_dim), np.float16)
    #         bert_answer_embeddings_batch = np.zeros((self.T_a_encoder, self.num_combinations, self.bert_dim), np.float16)
    #         if self.load_rationale:
    #             bert_rationale_embeddings_batch = np.zeros((self.T_r_encoder, self.num_combinations, self.bert_dim), np.float16)

    #     question_length_batch = np.zeros(self.num_combinations, np.int32)
    #     image_feat_batch = np.zeros(
    #         (self.num_combinations, self.feat_H, self.feat_W, self.feat_D),
    #         np.float32)

    #     image_path_list = [None]*self.num_combinations
    #     qid_list = [None]*self.num_combinations
    #     qstr_list = [None]*self.num_combinations
    #     sample_index = [None]*self.num_combinations
    #     answer_index = [None]*self.num_combinations
    #     all_answers_list = [None]*self.num_combinations
    #     all_answers_token_list = [None] * self.num_combinations
    #     if self.load_rationale:
    #         all_rationales_list = [None] * self.num_combinations
    #         all_rationales_token_list = [None] * self.num_combinations

    #     if self.load_correct_answer:
    #         answer_label_batch = np.zeros([self.num_combinations], np.float32)
    #         answer_onehot_batch = np.zeros([self.num_combinations], np.int32)
    #         if self.load_soft_score:
    #             num_choices = len(self.answer_dict.word_list)
    #             soft_score_batch = np.zeros(
    #                 (self.num_combinations, num_choices), np.float32)
        
    #     if self.load_correct_rationale:
    #         rationale_label_batch = np.zeros([self.num_combinations], np.float32)
    #         rationale_onehot_batch = np.zeros([self.num_combinations], np.int32)
    #         if self.load_soft_score:
    #             num_choices = len(self.answer_dict.word_list)
    #             soft_score_batch = np.zeros(
    #                 (self.num_combinations, num_choices), np.float32)
        
    #     if self.load_correct_answer and self.load_correct_rationale:
    #         answer_and_rationale_label_batch = np.zeros([self.num_combinations], np.float32)

    #     if self.load_gt_layout:
    #         gt_layout_question_batch = self.layout_dict.word2idx('_NoOp') * np.ones(
    #             (self.T_decoder, self.num_combinations), np.int32)

    #     # Precalculate the divisor and mod for later use. Sorry for the code mess.
    #     i_ans_divisor = self.num_combinations // self.num_answers
    #     i_rat_mod = self.num_combinations // self.num_rationales
        
    #     # Populate the arrays with each possible q-a pair.
    #     # Iterate over each sample,
    #     iminfo = self.imdb[sample_id]
    #     question_inds = [
    #         self.vocab_dict.word2idx(w) for w in iminfo['question_tokens']]

    #     all_answers = iminfo['all_answers']
    #     all_answers_tokens = [[self.vocab_dict.word2idx(w) for w in answer] for answer in all_answers]
    #     if self.load_rationale:
    #         all_rationales = iminfo['all_rationales']
    #         all_rationales_tokens = [[self.vocab_dict.word2idx(w) for w in rationale] for rationale in all_rationales]
    #     image_feat = np.load(iminfo['feature_path'])
    #     seq_length = min(len(question_inds), self.T_q_encoder)
        
    #     sample_range_in_batch = range(self.num_combinations)
    #     # Assign the question sequence to each column for each combination.
    #     question_seq_batch.T[sample_range_in_batch, :seq_length] = question_inds[:seq_length]
    #     question_length_batch[sample_range_in_batch] = seq_length

    #     if self.load_correct_answer:
    #         # Get the index of the correct answer choice.
    #         answer = iminfo['valid_answers'].index(0)
    #     if self.load_correct_rationale and self.load_rationale:
    #         # Get the index of the correct rationale choice.
    #         rationale = iminfo['valid_rationales'].index(0)

    #     if self.load_bert:
    #         # Look up the corresponding embeddings from the dataset.
    #         ans_embeddings = self.ans_hf[str(sample_id)]
    #         rat_embeddings = self.rat_hf[str(sample_id)]

    #         # Extract the context and answer/rationale embeddings.
    #         bert_ctx_answers, bert_answers = self.bert_handler.get_embeddings_from_group(ans_embeddings)
    #         bert_ctx_rationales, bert_rationales = self.bert_handler.get_embeddings_from_group(rat_embeddings)

    #         self.bert_handler.validate_embeddings(bert_ctx_answers, bert_ctx_rationales, bert_answers, bert_rationales, iminfo)

    #     for n, i in enumerate(sample_range_in_batch):
    #         i_ans = n // i_ans_divisor
    #         i_rat = n % i_rat_mod
    #         # The i:i+1 slice is necessary to unwrap the enclosing array of the image features.
    #         image_feat_batch[i:i+1] = image_feat
    #         image_path_list[i] = iminfo['image_path']
    #         qid_list[i] = iminfo['question_id']
    #         qstr_list[i] = iminfo['question_str']
    #         sample_index[i] = sample_id
    #         answer_index[i] = i_ans
    #         all_answers_list[i] = np.array(all_answers[i_ans])
    #         all_answers_token_list[i] = [all_answers_tokens[i_ans]]
    #         if self.load_rationale:
    #             all_rationales_list[i] = all_rationales[i_rat]
    #             all_rationales_token_list[i] = [all_rationales_tokens[i_rat]]

    #         if self.load_correct_answer:
    #             answer_label_batch[i] = 1. if answer == i_ans else 0.
    #             answer_onehot_batch[i] = answer_label_batch[i]

    #             # if self.load_soft_score:
    #             #     soft_score_inds = iminfo['soft_score_inds']
    #             #     soft_score_target = iminfo['soft_score_target']
    #             #     soft_score_batch[i_per_sample, soft_score_inds] = soft_score_target
            
    #         if self.load_correct_rationale and self.load_rationale:
    #             rationale_label_batch[i] = 1. if rationale == i_rat else 0.
    #             rationale_onehot_batch[i] = rationale_label_batch[i]

    #         if self.load_correct_answer and self.load_correct_rationale and self.load_rationale:
    #             answer_and_rationale_label_batch[i] = 1. if rationale == i_rat and answer == i_ans else 0.

    #             # if self.load_soft_score:
    #             #     soft_score_inds = iminfo['soft_score_inds']
    #             #     soft_score_target = iminfo['soft_score_target']
    #             #     soft_score_batch[i_per_sample, soft_score_inds] = soft_score_target

    #         # For each set of answers per-question, populate the list of supported answers in a sequence for embedding_lookup.
    #         for token_list in all_answers_token_list[i]:
    #             seq_length = min(len(token_list), self.T_a_encoder)
    #             all_answers_seq_batch[:seq_length, i] = token_list[:seq_length]
    #             all_answers_length_batch[i] = seq_length

    #         if self.load_rationale:
    #             # For each set of rationales per-question, populate the list of supported rationales in a sequence for embedding_lookup.
    #             for token_list in all_rationales_token_list[i]:
    #                 seq_length = min(len(token_list), self.T_r_encoder)
    #                 all_rationales_seq_batch[:seq_length, i] = token_list[:seq_length]
    #                 all_rationales_length_batch[i] = seq_length

    #         if self.load_bert:
    #             q_len = question_length_batch[i]
    #             a_len = all_answers_length_batch[i]
    #             if self.vcr_task_type == 'Q_2_A':
    #                 bert_question_embeddings_batch[:q_len, i] = bert_ctx_answers[i_ans][:q_len]
    #                 bert_answer_embeddings_batch[:a_len, i] = bert_answers[i_ans][:a_len]
    #             else:
    #                 r_len = all_rationales_length_batch[i]
    #                 assert True == False, 'Still have to finish this bit: Load all answer embeddings combos.'
    #                 if self.load_correct_answer:
    #                     # How to handle this part still confuses me cos of how we train rationales (all answer combinations).
    #                     bert_question_embeddings_batch[:q_len, i] = bert_ctx_rationales[n][:q_len]
    #                     bert_answer_embeddings_batch[:a_len, i] = bert_ctx_rationales[n][q_len:]
    #                     bert_rationale_embeddings_batch[:r_len, i] = bert_rationales[n]
    #                 else:
    #                     bert_question_embeddings_batch[:q_len, i] = bert_ctx_rationales[n][:q_len]
    #                     bert_answer_embeddings_batch[:a_len, i] = bert_ctx_rationales[n][q_len:a_len]
    #                     bert_rationale_embeddings_batch[:r_len, i] = bert_rationales[n]

    #     if self.load_gt_layout:
    #         # Get and load the gt layout for each question-answer available.
    #         gt_layout_qa_tokens_list = iminfo['gt_layout_qa_tokens']
    #         for n, i in enumerate(sample_range_in_batch):

    #             gt_layout_qa_tokens = gt_layout_qa_tokens_list[n]

    #             if self.prune_filter_module:
    #                 # remove duplicated consequtive modules
    #                 # (only keeping one _Filter)
    #                 for n_t in range(len(gt_layout_qa_tokens)-1, 0, -1):
    #                     if (gt_layout_qa_tokens[n_t-1] in {'_Filter', '_Find'}
    #                             and gt_layout_qa_tokens[n_t] == '_Filter'):
    #                         gt_layout_qa_tokens[n_t] = None
    #                 gt_layout_qa_tokens = [t for t in gt_layout_qa_tokens if t]

    #             question_layout_inds = [
    #                 self.layout_dict.word2idx(w) for w in gt_layout_qa_tokens]
    #             gt_layout_question_batch[:len(question_layout_inds), i] = question_layout_inds

    #     batch = dict(question_seq_batch=question_seq_batch,
    #                  question_length_batch=question_length_batch,
    #                  image_feat_batch=image_feat_batch,
    #                  answer_index=answer_index,
    #                  all_answers_list=all_answers_list,
    #                  all_answers_seq_batch=all_answers_seq_batch,
    #                  all_answers_length_batch=all_answers_length_batch,
    #                  qid_list=qid_list,
    #                  image_path_list=image_path_list
    #             )

    #     if self.load_rationale:
    #         batch.update(
    #             all_rationales_seq_batch=all_rationales_seq_batch,
    #             all_rationales_length_batch=all_rationales_length_batch,
    #         )

    #     if self.load_correct_answer:
    #         if self.data_params['use_sparse_softmax_labels'] == True:
    #             batch['answer_label_batch'] = np.where(np.reshape(answer_label_batch, (len(answer_label_batch) // self.num_combinations, self.num_combinations)) == 1.)[1]
    #         else:
    #             batch['answer_label_batch'] = answer_label_batch
    #         if self.load_soft_score:
    #             batch['soft_score_batch'] = soft_score_batch
    #     if self.load_correct_rationale and self.load_rationale:
    #         if self.data_params['use_sparse_softmax_labels'] == True:
    #             batch['rationale_label_batch'] = np.where(np.reshape(rationale_label_batch, (len(rationale_label_batch) // self.num_combinations, self.num_combinations)) == 1.)[1]
    #         else:
    #             batch['rationale_label_batch'] = rationale_label_batch
    #     if self.load_correct_answer and self.load_correct_rationale and self.load_rationale:
    #         if self.data_params['use_sparse_softmax_labels'] == True:
    #             batch['answer_and_rationale_label_batch'] = np.where(np.reshape(answer_and_rationale_label_batch, (len(answer_and_rationale_label_batch) // self.num_combinations, self.num_combinations)) == 1.)[1]
    #         else:
    #             batch['answer_and_rationale_label_batch'] = answer_and_rationale_label_batch
    #     if self.load_gt_layout:
    #         batch['gt_layout_question_batch'] = gt_layout_question_batch

    #     if self.load_bert:
    #         batch['bert_question_embeddings_batch'] = bert_question_embeddings_batch
    #         batch['bert_answer_embeddings_batch'] = bert_answer_embeddings_batch
    #         if self.load_rationale:
    #             batch['bert_rationale_embeddings_batch'] = bert_rationale_embeddings_batch

    #     return batch

    def __del__(self):
        if hasattr(self, 'bert_handler'):
            del bert_handler

    def get_per_device_print_fn(self, tensor) -> Callable[[str], None]:
        device = tensor.device
        return lambda msg: print(f'{device + ": " if device is not None else ""}DataReader: {msg}')
