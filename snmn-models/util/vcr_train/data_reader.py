import threading
import queue
import numpy as np

from util import text_processing


class BatchLoaderVcr:
    def __init__(self, imdb, data_params):
        self.imdb = imdb
        self.data_params = data_params
        self.vcr_task_type = data_params['vcr_task_type']

        self.vocab_dict = text_processing.VocabDict(
            data_params['vocab_question_file'], first_token_only=True)
        self.T_encoder = data_params['T_encoder']

        # peek one example to see whether answer and gt_layout are in the data
        self.load_answer = (
            'valid_answers' in self.imdb[0])
        self.load_rationale = (
            'valid_rationales' in self.imdb[0])
        self.load_gt_layout = (
            ('load_gt_layout' in data_params and data_params['load_gt_layout'])
            and ('gt_layout_qa_tokens' in self.imdb[0] and
                 self.imdb[0]['gt_layout_qa_tokens'] is not None))

        self.num_answers = len(self.imdb[0]['all_answers'])
        self.num_rationales = len(self.imdb[0]['all_rationales'])

        self.num_combinations = self.num_answers * self.num_rationales

        if not self.load_answer:
            print('imdb does not contain answers')
        if not self.load_rationale:
            print('imdb does not contain rationales')

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

    def load_one_batch(self, sample_ids):
        if self.vcr_task_type == 'Q_2_AR' or self.vcr_task_type == 'QA_2_R':
            actual_batch_size = len(sample_ids) * self.num_answers * self.num_rationales
        else:
            actual_batch_size = len(sample_ids) * self.num_answers

        per_answer_size = 1

        # Allocate the arrays and collections.

        question_seq_batch = np.zeros(
            (self.T_encoder, actual_batch_size), np.int32)
        all_answers_seq_batch = np.zeros((per_answer_size, self.T_encoder, actual_batch_size), np.int32)
        all_answers_length_batch = np.zeros((per_answer_size, actual_batch_size), np.int32)
        all_rationales_seq_batch = np.zeros((per_answer_size, self.T_encoder, actual_batch_size), np.int32)
        all_rationales_length_batch = np.zeros((per_answer_size, actual_batch_size), np.int32)

        question_length_batch = np.zeros(actual_batch_size, np.int32)
        image_feat_batch = np.zeros(
            (actual_batch_size, self.feat_H, self.feat_W, self.feat_D),
            np.float32)
        image_path_list = [None]*actual_batch_size
        qid_list = [None]*actual_batch_size
        qstr_list = [None]*actual_batch_size
        all_answers_list = [None]*actual_batch_size
        all_answers_token_list = [None] * actual_batch_size
        all_rationales_list = [None] * actual_batch_size
        all_rationales_token_list = [None] * actual_batch_size

        if self.load_answer:
            answer_label_batch = np.zeros([actual_batch_size, 1], np.float32)
            answer_onehot_batch = np.zeros([actual_batch_size, per_answer_size], np.int32)
            valid_answers_list = [None]*actual_batch_size
            if self.load_soft_score:
                num_choices = len(self.answer_dict.word_list)
                soft_score_batch = np.zeros(
                    (actual_batch_size, num_choices), np.float32)
        
        if self.load_rationale:
            rationale_label_batch = np.zeros([actual_batch_size, 1], np.float32)
            rationale_onehot_batch = np.zeros([actual_batch_size, per_answer_size], np.int32)
            valid_rationales_list = [None]*actual_batch_size
            if self.load_soft_score:
                num_choices = len(self.answer_dict.word_list)
                soft_score_batch = np.zeros(
                    (actual_batch_size, num_choices), np.float32)
        
        if self.load_answer and self.load_rationale:
            answer_and_rationale_label_batch = np.zeros([actual_batch_size, 1], np.float32)

        if self.load_gt_layout:
            gt_layout_question_batch = self.layout_dict.word2idx('_NoOp') * np.ones(
                (self.T_decoder, actual_batch_size), np.int32)

        # Populate the arrays with each possible q-a pair.

        # Iterate over each sample,
        # for i_per_sample, i_per_answer, i_per_rationale in zip(range(len(sample_ids)), range(0, actual_batch_size, self.num_answers * self.num_rationales), range(0, actual_batch_size, self.num_answers * self.num_rationales)):
        combinations_per_sample = self.num_answers * self.num_rationales if self.load_rationale else self.num_answers

        for i_per_sample, i_per_qar in zip(range(len(sample_ids)), range(0, actual_batch_size, combinations_per_sample)):
            iminfo = self.imdb[sample_ids[i_per_sample]]
            question_inds = [
                self.vocab_dict.word2idx(w) for w in iminfo['question_tokens']]

            all_answers = iminfo['all_answers']
            all_answers_tokens = [[self.vocab_dict.word2idx(w) for w in answer] for answer in all_answers]
            all_rationales = iminfo['all_rationales']
            all_rationales_tokens = [[self.vocab_dict.word2idx(w) for w in rationale] for rationale in all_rationales]
            image_feat = np.load(iminfo['feature_path'])
            seq_length = len(question_inds)
            question_seq_batch[:seq_length, i_per_qar] = question_inds

            if self.load_answer:
                # Get the index of the correct answer choice.
                answer = iminfo['valid_answers'].index(0)
            if self.load_rationale:
                # Get the index of the correct rationale choice.
                rationale = iminfo['valid_rationales'].index(0)

            sample_range_in_batch = range(i_per_qar, i_per_qar + combinations_per_sample)
            for n, i in enumerate(sample_range_in_batch):
                i_ans = n // self.num_answers
                i_rat = n % self.num_answers
                question_length_batch[i] = seq_length
                # The i:i+1 slice is necessary to unwrap the enclosing array of the image features.
                image_feat_batch[i:i+1] = image_feat
                image_path_list[i] = iminfo['image_path']
                qid_list[i] = iminfo['question_id']
                qstr_list[i] = iminfo['question_str']
                all_answers_list[i] = all_answers[i_ans]
                all_answers_token_list[i] = [all_answers_tokens[i_ans]]
                all_rationales_list[i] = all_rationales[i_rat]
                all_rationales_token_list[i] = [all_rationales_tokens[i_rat]]

                if self.load_answer:
                    answer_label_batch[i] = [1. if answer == i_ans else 0.]
                    answer_onehot_batch[i] = answer_label_batch[i][0]

                    # if self.load_soft_score:
                    #     soft_score_inds = iminfo['soft_score_inds']
                    #     soft_score_target = iminfo['soft_score_target']
                    #     soft_score_batch[i_per_sample, soft_score_inds] = soft_score_target
                
                if self.load_rationale:
                    rationale_label_batch[i] = [1. if rationale == i_rat else 0.]
                    rationale_onehot_batch[i] = rationale_label_batch[i][0]

                if self.load_answer and self.load_rationale:
                    answer_and_rationale_label_batch[i] = [1. if rationale == i_rat and answer == i_ans else 0.]

                    # if self.load_soft_score:
                    #     soft_score_inds = iminfo['soft_score_inds']
                    #     soft_score_target = iminfo['soft_score_target']
                    #     soft_score_batch[i_per_sample, soft_score_inds] = soft_score_target

                # For each set of answers per-question, populate the list of supported answers in a sequence for embedding_lookup.
                for i_token, token_list in enumerate(all_answers_token_list[i]):
                    seq_length = len(token_list)
                    all_answers_seq_batch[i_token, :seq_length, i] = token_list
                    all_answers_length_batch[i_token, i] = seq_length
                # For each set of rationales per-question, populate the list of supported rationales in a sequence for embedding_lookup.
                for i_token, token_list in enumerate(all_rationales_token_list[i]):
                    seq_length = len(token_list)
                    all_rationales_seq_batch[i_token, :seq_length, i] = token_list
                    all_rationales_length_batch[i_token, i] = seq_length

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
                     image_path_list=image_path_list,
                     qid_list=qid_list,
                     qstr_list=qstr_list,
                     all_answers_list=all_answers_list,
                     all_answers_token_list=all_answers_token_list,
                     all_answers_seq_batch=all_answers_seq_batch,
                     all_answers_length_batch=all_answers_length_batch,
                     answer_onehot_batch=answer_onehot_batch,
                     all_rationales_list=all_rationales_list,
                     all_rationales_token_list=all_rationales_token_list,
                     all_rationales_seq_batch=all_rationales_seq_batch,
                     all_rationales_length_batch=all_rationales_length_batch,
                     rationale_onehot_batch=rationale_onehot_batch,)

        if self.load_answer:
            batch['answer_label_batch'] = answer_label_batch
            if self.load_soft_score:
                batch['soft_score_batch'] = soft_score_batch
        if self.load_rationale:
            batch['rationale_label_batch'] = rationale_label_batch
        if self.load_answer and self.load_rationale:
            batch['answer_and_rationale_label_batch'] = answer_and_rationale_label_batch
        if self.load_gt_layout:
            batch['gt_layout_question_batch'] = gt_layout_question_batch

        return batch


class DataReader:
    def __init__(self, imdb_file, shuffle=True, one_pass=False, prefetch_num=8,
                 **kwargs):
        print('Loading imdb from %s' % imdb_file)
        if imdb_file.endswith('.npy'):
            imdb = np.load(imdb_file, allow_pickle=True)
        else:
            raise TypeError('unknown imdb format.')
        print('Done')
        self.imdb = imdb
        self.shuffle = shuffle
        self.one_pass = one_pass
        self.prefetch_num = prefetch_num
        self.data_params = kwargs

        # Vqa data loader
        self.batch_loader = BatchLoaderVcr(self.imdb, self.data_params)

        # Start prefetching thread
        self.prefetch_queue = queue.Queue(maxsize=self.prefetch_num)
        self.prefetch_thread = threading.Thread(
            target=_run_prefetch, args=(
                self.prefetch_queue, self.batch_loader, self.imdb,
                self.shuffle, self.one_pass, self.data_params))
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

    def batches(self):
        while True:
            # Get a batch from the prefetching queue
            # if self.prefetch_queue.empty():
            #     print('data reader: waiting for data loading...')
            batch = self.prefetch_queue.get(block=True)
            if batch is None:
                assert(self.one_pass)
                print('data reader: one pass finished')
                raise StopIteration()
            yield batch


def _run_prefetch(prefetch_queue, batch_loader, imdb, shuffle, one_pass,
                  data_params):
    num_samples = len(imdb)
    batch_size = data_params['batch_size']

    n_sample = 0
    fetch_order = np.arange(num_samples)
    while True:
        # Shuffle the sample order for every epoch
        if n_sample == 0 and shuffle:
            fetch_order = np.random.permutation(num_samples)

        # Load batch from file
        # note that len(sample_ids) <= batch_size, not necessarily equal
        sample_ids = fetch_order[n_sample:n_sample+batch_size]
        batch = batch_loader.load_one_batch(sample_ids)
        prefetch_queue.put(batch, block=True)

        n_sample += len(sample_ids)
        if n_sample >= num_samples:
            # Put in a None batch to indicate a whole pass is over
            if one_pass:
                prefetch_queue.put(None, block=True)
            n_sample = 0
