import h5py
import numpy as np
import tensorflow as tf

from util import text_processing


class DataReader:
    def __init__(self, imdb_file, **data_params):
        print('Loading imdb from %s' % imdb_file)
        if imdb_file.endswith('.npy'):
            imdb = np.load(imdb_file, allow_pickle=True)
        else:
            raise TypeError('unknown imdb format.')
        print('Done')

        self.imdb = imdb
        self.data_params = data_params
        self.imdb = imdb
        self.data_params = data_params
        self.vcr_task_type = data_params['vcr_task_type']

        self.vocab_dict = text_processing.VocabDict(
            data_params['vocab_question_file'], first_token_only=True)
        self.T_q_encoder = data_params['T_q_encoder']
        self.T_a_encoder = data_params['T_a_encoder']
        self.T_r_encoder = data_params['T_r_encoder']

        if data_params['load_bert_embeddings'] == True:
            print('Loading BERT embeddings.')
            self.load_bert = True
            self.ans_hf = h5py.File(self.data_params['bert_answer_embeddings_path'], mode='r')
            self.rat_hf = h5py.File(self.data_params['bert_rationale_embeddings_path'], mode='r')
            self.bert_dim = len(self.ans_hf['0']['answer_answer0'][0])
        else:
            self.load_bert = False

        # peek one example to see whether answer and gt_layout are in the data
        self.load_correct_answer = (
            'valid_answers' in self.imdb[0])
        self.load_correct_rationale = (
            'valid_rationales' in self.imdb[0])
        self.load_gt_layout = (
            ('load_gt_layout' in data_params and data_params['load_gt_layout'])
            and ('gt_layout_qa_tokens' in self.imdb[0] and
                 self.imdb[0]['gt_layout_qa_tokens'] is not None))
        self.load_rationale = data_params['vcr_task_type'] == 'QA_2_R' or data_params['vcr_task_type'] == 'Q_2_AR'

        self.num_answers = len(self.imdb[0]['all_answers'])
        self.num_rationales = len(self.imdb[0]['all_rationales'])

        if self.data_params['vcr_task_type'] == 'Q_2_A':
            self.correct_label_batch_name = 'answer_label_batch'
        elif self.data_params['vcr_task_type'] == 'QA_2_R':
            self.correct_label_batch_name = 'rationale_label_batch'
        elif self.data_params['vcr_task_type'] == 'Q_2_AR':
            self.correct_label_batch_name = 'answer_and_rationale_label_batch'

        if data_params['vcr_task_type'] == 'Q_2_A':
            self.num_combinations = self.num_answers
        elif data_params['vcr_task_type'] == 'QA_2_R':
            self.num_combinations = self.num_rationales
        else:
            self.num_combinations = self.num_answers * self.num_rationales

        self.actual_batch_size = data_params['batch_size'] * self.num_combinations

        if not self.load_correct_answer:
            print('imdb does not contain correct answers')
        if not self.load_correct_rationale:
            print('imdb does not contain correct rationales')

        if not self.load_rationale:
            print('Model is running in neither QA_2_R nor Q_2_AR')

        self.T_decoder = data_params['T_decoder']
        self.layout_dict = text_processing.VocabDict(
            data_params['vocab_layout_file'])
        if self.load_gt_layout:
            # Prune multiple filter modules by default
            self.prune_filter_module = (
                data_params['prune_filter_module']
                if 'prune_filter_module' in data_params else True)
        else:
            print('imdb does not contain ground-truth layout')
        # Whether to load soft scores (targets for sigmoid regression)
        self.load_soft_score = ('load_soft_score' in data_params and
                                data_params['load_soft_score'])

        # load one feature map to peek its size
        feats = np.load(self.imdb[0]['feature_path'])
        self.feat_H, self.feat_W, self.feat_D = feats.shape[1:]

        self.output_types = {
            'image_feat_batch': tf.float32,
            'question_seq_batch': tf.int32,
            'question_length_batch': tf.int32,
            'answer_index': tf.int32,
            'all_answers_seq_batch': tf.int32,
            'all_answers_length_batch': tf.int32,
            'image_path_list': tf.string,
            'qid_list': tf.int32
        }
        if self.load_correct_answer:
            self.output_types['answer_label_batch'] = tf.int32 if self.data_params['use_sparse_softmax_labels'] else tf.float32
        if self.load_rationale:
            if self.load_correct_rationale:
                self.output_types['rationale_label_batch'] = tf.int32 if self.data_params['use_sparse_softmax_labels'] else tf.float32
            self.output_types['rationale_seq_batch'] = tf.int32
            self.output_types['rationale_length_batch'] = tf.int32
        if self.load_bert:
            self.output_types['bert_question_embeddings_batch'] = tf.float16
            self.output_types['bert_answer_embeddings_batch'] = tf.float16
            if self.load_rationale:
                self.output_types['bert_rationale_embeddings_batch'] = tf.float16

        self.output_shapes = {
            'image_feat_batch': tf.TensorShape([self.feat_H, self.feat_W, self.feat_D]),
            'question_seq_batch': tf.TensorShape([None]),
            'question_length_batch': tf.TensorShape([]),
            'answer_index': tf.TensorShape([]),
            'all_answers_seq_batch': tf.TensorShape([None]),
            'all_answers_length_batch': tf.TensorShape([]),
            'image_path_list': tf.TensorShape([]),
            'qid_list': tf.TensorShape([])
        }
        if self.load_correct_answer:
            self.output_shapes['answer_label_batch'] = tf.TensorShape([])
        if self.load_rationale:
            if self.load_correct_rationale:
                self.output_shapes['rationale_label_batch'] = tf.TensorShape([])
            self.output_shapes['rationale_seq_batch'] = tf.TensorShape([None])
            self.output_shapes['rationale_length_batch'] = tf.TensorShape([])
        if self.load_bert:
            self.output_shapes['bert_question_embeddings_batch'] = tf.TensorShape([None, self.bert_dim])
            self.output_shapes['bert_answer_embeddings_batch'] = tf.TensorShape([None, self.bert_dim])
            if self.load_rationale:
                self.output_shapes['bert_rationale_embeddings_batch'] = tf.TensorShape([None, self.bert_dim])

    def init_dataset(self):
        print('Initialising dataset.')

        # Vqa data loader
        dataset: tf.compat.v1.data.Dataset = tf.compat.v1.data.Dataset.from_generator(self.batches, self.output_types, self.output_shapes).batch(self.actual_batch_size)
        self.prefetch_size=64 if 'prefetch_size' not in self.data_params else self.data_params['prefetch_size']

        print(f'Using unshuffled prefetch of size {self.prefetch_size}.')
        dataset.prefetch(buffer_size=self.prefetch_size)
        # if not self.data_params['use_sparse_softmax_labels']:
        #     print('Sparse softmax labels disabled. Enabling dataset shuffling.')
        #     self.dataset = self.dataset.shuffle(buffer_size=32)
        # else:
        #     print('Sparse softmax labels enabled. Using unshuffled prefetch instead.')
        #     self.dataset =self.dataset.prefetch(buffer_size=32)
        
        dataset = dataset.map(self.to_time_major)
        dataset = dataset.map(self.name_tensors)
        
        return dataset
    
    def name_tensors(self, element):
        for key in element.keys():
            element[key] = tf.identity(element[key], name=key)
        return element

    def to_time_major(self, element):
        element['question_seq_batch'] = tf.transpose(element['question_seq_batch'])
        element['all_answers_seq_batch'] = tf.transpose(element['all_answers_seq_batch'])
        if self.load_rationale:
            element['all_rationales_seq_batch'] = tf.transpose(element['all_rationales_seq_batch'])
        if self.load_bert:
            element['bert_question_embeddings_batch'] = tf.transpose(element['bert_question_embeddings_batch'], [1, 0, 2])
            element['bert_answer_embeddings_batch'] = tf.transpose(element['bert_answer_embeddings_batch'], [1, 0, 2])
            if self.load_rationale:
                element['bert_rationale_embeddings_batch'] = tf.transpose(element['bert_rationale_embeddings_batch'], [1, 0, 2])
        return element

    def flatten_batch_task_dims(self, element):
        for key in element.keys():
            # Get the shape of the current tensor
            shape = tf.shape(element[key])
        
            # Since each element contains n records (1 for each combination of answer/rationale/answer-rationale), flatten that dimension into the batch_size dim.
            # The new size effectively becomes self.num_combinations * self.batch_size
            new_shape = tf.concat(([shape[0] * shape[1]], shape[2:]), axis=0)
        
            # Reshape the current tensor and update it in the dictionary
            element[key] = tf.reshape(element[key], new_shape)
        return element

    def batches(self):
        num_samples = len(self.imdb)
        for sample_id in np.random.permutation(num_samples):
            collection = self.load_one(sample_id)

            for i in range(self.num_combinations):
                record = {}
                record['image_feat_batch'] = collection['image_feat_batch'][i]
                record['question_seq_batch'] = collection['question_seq_batch'][:, i]
                record['question_length_batch'] = collection['question_length_batch'][i]
                record['answer_index'] = collection['answer_index'][i]
                record['all_answers_seq_batch'] = collection['all_answers_seq_batch'][:, i]
                record['all_answers_length_batch'] = collection['all_answers_length_batch'][i]
                record['image_path_list'] = collection['image_path_list'][i]
                record['qid_list'] = collection['qid_list'][i]
                if self.load_correct_answer:
                    record['answer_label_batch'] = collection['answer_label_batch'][i]
                if self.load_rationale:
                    record['all_rationales_seq_batch'] = collection['all_rationales_seq_batch'][:, i]
                    record['all_rationales_length_batch'] = collection['all_rationales_length_batch'][i]
                    if self.load_correct_answer:
                        record['rationale_label_batch'] = collection['rationale_label_batch'][i]
                if self.load_bert:
                    record['bert_question_embeddings_batch'] = collection['bert_question_embeddings_batch'][:, i]
                    record['bert_answer_embeddings_batch'] = collection['bert_answer_embeddings_batch'][:, i]
                    if self.load_rationale:
                        record['bert_rationale_embeddings_batch'] = collection['bert_rationale_embeddings_batch'][:, i]
                yield record

    def get_embeddings_from_group(self, hgroup):
        ans = []
        ctx = []

        for subkey, dataset in hgroup.items():
            if subkey.startswith('answer_'):
                if subkey.startswith('answer_answer') or subkey.startswith('answer_rationale'):
                    ans.append(np.array(dataset, np.float16))
                else:
                    raise ValueError(f'Unexpected key {subkey}')
            elif subkey.startswith('ctx_'):
                ctx.append(np.array(dataset, np.float16))
            else:
                raise ValueError(f'Unexpected key {subkey}')
        return ctx, ans

    def validate_embeddings(self, ctx_answers, ctx_rationales, answers, rationales, qar):
        if not self.load_correct_answer:
            assert len(ctx_rationales) == len(qar['all_answers']) * len(qar['all_rationales']), 'Not all combinations of answers and rationales were found.'
        else:
            assert np.shape(ctx_answers)[2] == np.shape(ctx_rationales)[2] and np.shape(ctx_answers)[0] == np.shape(ctx_rationales)[0], 'Shapes of answer and rationale contexts do not match.'

        assert np.shape(ctx_answers)[1] == len(qar['question_tokens']), 'Shapes of answer contexts do not match length of question.'

        for a, qar_a in zip(answers, qar['all_answers']):
            assert len(a) == len(qar_a), f'Answer pairing {str(a)} and {str(qar_a)} don\'t match.'

        for j, (ctx, r),  in enumerate(zip(ctx_rationales, rationales)):
            rat_i = j % len(qar['all_rationales'])
            if not self.load_correct_answer:
                ans_i = int(j // len(qar['all_rationales']))
                assert len(ctx) == len(qar['question_tokens']) + len(qar['all_answers'][ans_i]), 'Shapes of rationale contexts do not match length of question and correct answer.'
            else:
                assert len(ctx) == len(qar['question_tokens']) + len(qar['all_answers'][qar['valid_answer_index']]), 'Shapes of rationale contexts do not match length of question and correct answer.'
            assert len(r) == len(qar['all_rationales'][rat_i]), f'Rationale pairing {str(r)} and {str(qar["all_rationales"][rat_i])} don\'t match.'

    def load_one(self, sample_id):

        # Allocate the arrays and collections.

        question_seq_batch = np.zeros(
            (self.T_q_encoder, self.num_combinations), np.int32)
        all_answers_seq_batch = np.zeros((self.T_a_encoder, self.num_combinations), np.int32)
        all_answers_length_batch = np.zeros((self.num_combinations), np.int32)
        if self.load_rationale:
            all_rationales_seq_batch = np.zeros((self.T_r_encoder, self.num_combinations), np.int32)
            all_rationales_length_batch = np.zeros((self.num_combinations), np.int32)
        if self.load_bert:
            bert_question_embeddings_batch = np.zeros((self.T_q_encoder, self.num_combinations, self.bert_dim), np.float16)
            bert_answer_embeddings_batch = np.zeros((self.T_a_encoder, self.num_combinations, self.bert_dim), np.float16)
            if self.load_rationale:
                bert_rationale_embeddings_batch = np.zeros((self.T_r_encoder, self.num_combinations, self.bert_dim), np.float16)

        question_length_batch = np.zeros(self.num_combinations, np.int32)
        image_feat_batch = np.zeros(
            (self.num_combinations, self.feat_H, self.feat_W, self.feat_D),
            np.float32)

        image_path_list = [None]*self.num_combinations
        qid_list = [None]*self.num_combinations
        qstr_list = [None]*self.num_combinations
        sample_index = [None]*self.num_combinations
        answer_index = [None]*self.num_combinations
        all_answers_list = [None]*self.num_combinations
        all_answers_token_list = [None] * self.num_combinations
        if self.load_rationale:
            all_rationales_list = [None] * self.num_combinations
            all_rationales_token_list = [None] * self.num_combinations

        if self.load_correct_answer:
            answer_label_batch = np.zeros([self.num_combinations], np.float32)
            answer_onehot_batch = np.zeros([self.num_combinations], np.int32)
            if self.load_soft_score:
                num_choices = len(self.answer_dict.word_list)
                soft_score_batch = np.zeros(
                    (self.num_combinations, num_choices), np.float32)
        
        if self.load_correct_rationale:
            rationale_label_batch = np.zeros([self.num_combinations], np.float32)
            rationale_onehot_batch = np.zeros([self.num_combinations], np.int32)
            if self.load_soft_score:
                num_choices = len(self.answer_dict.word_list)
                soft_score_batch = np.zeros(
                    (self.num_combinations, num_choices), np.float32)
        
        if self.load_correct_answer and self.load_correct_rationale:
            answer_and_rationale_label_batch = np.zeros([self.num_combinations], np.float32)

        if self.load_gt_layout:
            gt_layout_question_batch = self.layout_dict.word2idx('_NoOp') * np.ones(
                (self.T_decoder, self.num_combinations), np.int32)

        # Precalculate the divisor and mod for later use. Sorry for the code mess.
        i_ans_divisor = self.num_combinations // self.num_answers
        i_rat_mod = self.num_combinations // self.num_rationales
        
        # Populate the arrays with each possible q-a pair.
        # Iterate over each sample,
        iminfo = self.imdb[sample_id]
        question_inds = [
            self.vocab_dict.word2idx(w) for w in iminfo['question_tokens']]

        all_answers = iminfo['all_answers']
        all_answers_tokens = [[self.vocab_dict.word2idx(w) for w in answer] for answer in all_answers]
        if self.load_rationale:
            all_rationales = iminfo['all_rationales']
            all_rationales_tokens = [[self.vocab_dict.word2idx(w) for w in rationale] for rationale in all_rationales]
        image_feat = np.load(iminfo['feature_path'])
        seq_length = min(len(question_inds), self.T_q_encoder)
        
        sample_range_in_batch = range(self.num_combinations)
        # Assign the question sequence to each column for each combination.
        question_seq_batch.T[sample_range_in_batch, :seq_length] = question_inds[:seq_length]
        question_length_batch[sample_range_in_batch] = seq_length

        if self.load_correct_answer:
            # Get the index of the correct answer choice.
            answer = iminfo['valid_answers'].index(0)
        if self.load_correct_rationale and self.load_rationale:
            # Get the index of the correct rationale choice.
            rationale = iminfo['valid_rationales'].index(0)

        if self.load_bert:
            # Look up the corresponding embeddings from the dataset.
            ans_embeddings = self.ans_hf[str(sample_id)]
            rat_embeddings = self.rat_hf[str(sample_id)]

            # Extract the context and answer/rationale embeddings.
            bert_ctx_answers, bert_answers = self.get_embeddings_from_group(ans_embeddings)
            bert_ctx_rationales, bert_rationales = self.get_embeddings_from_group(rat_embeddings)

            self.validate_embeddings(bert_ctx_answers, bert_ctx_rationales, bert_answers, bert_rationales, iminfo)

        for n, i in enumerate(sample_range_in_batch):
            i_ans = n // i_ans_divisor
            i_rat = n % i_rat_mod
            # The i:i+1 slice is necessary to unwrap the enclosing array of the image features.
            image_feat_batch[i:i+1] = image_feat
            image_path_list[i] = iminfo['image_path']
            qid_list[i] = iminfo['question_id']
            qstr_list[i] = iminfo['question_str']
            sample_index[i] = sample_id
            answer_index[i] = i_ans
            all_answers_list[i] = np.array(all_answers[i_ans])
            all_answers_token_list[i] = [all_answers_tokens[i_ans]]
            if self.load_rationale:
                all_rationales_list[i] = all_rationales[i_rat]
                all_rationales_token_list[i] = [all_rationales_tokens[i_rat]]

            if self.load_correct_answer:
                answer_label_batch[i] = 1. if answer == i_ans else 0.
                answer_onehot_batch[i] = answer_label_batch[i]

                # if self.load_soft_score:
                #     soft_score_inds = iminfo['soft_score_inds']
                #     soft_score_target = iminfo['soft_score_target']
                #     soft_score_batch[i_per_sample, soft_score_inds] = soft_score_target
            
            if self.load_correct_rationale and self.load_rationale:
                rationale_label_batch[i] = 1. if rationale == i_rat else 0.
                rationale_onehot_batch[i] = rationale_label_batch[i]

            if self.load_correct_answer and self.load_correct_rationale and self.load_rationale:
                answer_and_rationale_label_batch[i] = 1. if rationale == i_rat and answer == i_ans else 0.

                # if self.load_soft_score:
                #     soft_score_inds = iminfo['soft_score_inds']
                #     soft_score_target = iminfo['soft_score_target']
                #     soft_score_batch[i_per_sample, soft_score_inds] = soft_score_target

            # For each set of answers per-question, populate the list of supported answers in a sequence for embedding_lookup.
            for token_list in all_answers_token_list[i]:
                seq_length = min(len(token_list), self.T_a_encoder)
                all_answers_seq_batch[:seq_length, i] = token_list[:seq_length]
                all_answers_length_batch[i] = seq_length

            if self.load_rationale:
                # For each set of rationales per-question, populate the list of supported rationales in a sequence for embedding_lookup.
                for token_list in all_rationales_token_list[i]:
                    seq_length = min(len(token_list), self.T_r_encoder)
                    all_rationales_seq_batch[:seq_length, i] = token_list[:seq_length]
                    all_rationales_length_batch[i] = seq_length

            if self.load_bert:
                q_len = question_length_batch[i]
                a_len = all_answers_length_batch[i]
                if self.vcr_task_type == 'Q_2_A':
                    bert_question_embeddings_batch[:q_len, i] = bert_ctx_answers[i_ans][:q_len]
                    bert_answer_embeddings_batch[:a_len, i] = bert_answers[i_ans][:a_len]
                else:
                    r_len = all_rationales_length_batch[i]
                    assert True == False, 'Still have to finish this bit: Load all answer embeddings combos.'
                    if self.load_correct_answer:
                        # How to handle this part still confuses me cos of how we train rationales (all answer combinations).
                        bert_question_embeddings_batch[:q_len, i] = bert_ctx_rationales[n][:q_len]
                        bert_answer_embeddings_batch[:a_len, i] = bert_ctx_rationales[n][q_len:]
                        bert_rationale_embeddings_batch[:r_len, i] = bert_rationales[n]
                    else:
                        bert_question_embeddings_batch[:q_len, i] = bert_ctx_rationales[n][:q_len]
                        bert_answer_embeddings_batch[:a_len, i] = bert_ctx_rationales[n][q_len:a_len]
                        bert_rationale_embeddings_batch[:r_len, i] = bert_rationales[n]

        if self.load_gt_layout:
            # Get and load the gt layout for each question-answer available.
            gt_layout_qa_tokens_list = iminfo['gt_layout_qa_tokens']
            for n, i in enumerate(sample_range_in_batch):

                gt_layout_qa_tokens = gt_layout_qa_tokens_list[n]

                if self.prune_filter_module:
                    # remove duplicated consequtive modules
                    # (only keeping one _Filter)
                    for n_t in range(len(gt_layout_qa_tokens)-1, 0, -1):
                        if (gt_layout_qa_tokens[n_t-1] in {'_Filter', '_Find'}
                                and gt_layout_qa_tokens[n_t] == '_Filter'):
                            gt_layout_qa_tokens[n_t] = None
                    gt_layout_qa_tokens = [t for t in gt_layout_qa_tokens if t]

                question_layout_inds = [
                    self.layout_dict.word2idx(w) for w in gt_layout_qa_tokens]
                gt_layout_question_batch[:len(question_layout_inds), i] = question_layout_inds

        batch = dict(question_seq_batch=question_seq_batch,
                     question_length_batch=question_length_batch,
                     image_feat_batch=image_feat_batch,
                     answer_index=answer_index,
                     all_answers_list=all_answers_list,
                     all_answers_seq_batch=all_answers_seq_batch,
                     all_answers_length_batch=all_answers_length_batch,
                     qid_list=qid_list,
                     image_path_list=image_path_list
                )

        if self.load_rationale:
            batch.update(
                all_rationales_seq_batch=all_rationales_seq_batch,
                all_rationales_length_batch=all_rationales_length_batch,
            )

        if self.load_correct_answer:
            if self.data_params['use_sparse_softmax_labels'] == True:
                batch['answer_label_batch'] = np.where(np.reshape(answer_label_batch, (len(answer_label_batch) // self.num_combinations, self.num_combinations)) == 1.)[1]
            else:
                batch['answer_label_batch'] = answer_label_batch
            if self.load_soft_score:
                batch['soft_score_batch'] = soft_score_batch
        if self.load_correct_rationale and self.load_rationale:
            if self.data_params['use_sparse_softmax_labels'] == True:
                batch['rationale_label_batch'] = np.where(np.reshape(rationale_label_batch, (len(rationale_label_batch) // self.num_combinations, self.num_combinations)) == 1.)[1]
            else:
                batch['rationale_label_batch'] = rationale_label_batch
        if self.load_correct_answer and self.load_correct_rationale and self.load_rationale:
            if self.data_params['use_sparse_softmax_labels'] == True:
                batch['answer_and_rationale_label_batch'] = np.where(np.reshape(answer_and_rationale_label_batch, (len(answer_and_rationale_label_batch) // self.num_combinations, self.num_combinations)) == 1.)[1]
            else:
                batch['answer_and_rationale_label_batch'] = answer_and_rationale_label_batch
        if self.load_gt_layout:
            batch['gt_layout_question_batch'] = gt_layout_question_batch

        if self.load_bert:
            batch['bert_question_embeddings_batch'] = bert_question_embeddings_batch
            batch['bert_answer_embeddings_batch'] = bert_answer_embeddings_batch
            if self.load_rationale:
                batch['bert_rationale_embeddings_batch'] = bert_rationale_embeddings_batch

        return batch
