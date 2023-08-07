import os
import json
import numpy as np
import tensorflow as tf
import itertools

from models_vcr.model import Model
from models_vcr.config import build_cfg_from_argparse
from util.vcr_train.data_reader import DataReader

# Load config
cfg = build_cfg_from_argparse()

# Start session
if os.environ["CUDA_VISIBLE_DEVICES"] is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPU_ID)
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# TODO: Enable XLA graph optimisations
tf.config.optimizer.set_jit('autoclustering')

sess = tf.Session(config=tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=cfg.GPU_MEM_GROWTH)))

batch_size=cfg.TRAIN.BATCH_SIZE

# Data files
imdb_file = cfg.IMDB_FILE % cfg.TEST.SPLIT_VQA
data_reader = DataReader(
    imdb_file, shuffle=True, one_pass=True, batch_size=batch_size,
    vocab_question_file=cfg.VOCAB_QUESTION_FILE,
    T_q_encoder=cfg.MODEL.T_Q_ENCODER,
    T_a_encoder=cfg.MODEL.T_A_ENCODER,
    T_r_encoder=cfg.MODEL.T_R_ENCODER,
    vocab_answer_file=cfg.VOCAB_ANSWER_FILE % cfg.TRAIN.SPLIT_VQA,
    load_gt_layout=cfg.TRAIN.USE_GT_LAYOUT,
    vocab_layout_file=cfg.VOCAB_LAYOUT_FILE, T_decoder=cfg.MODEL.T_CTRL,
    load_soft_score=cfg.TRAIN.VQA_USE_SOFT_SCORE,
    feed_answers_with_input=cfg.MODEL.INPUT.USE_ANSWERS,
    vcr_task_type=cfg.MODEL.VCR_TASK_TYPE,
    use_sparse_softmax_labels=cfg.TRAIN.SOLVER.USE_SPARSE_SOFTMAX_LABELS,
    load_bert_embeddings = cfg.USE_BERT_SENTENCE_EMBED,
    bert_answer_embeddings_path = cfg.BERT_EMBED_FILE % ('answer', cfg.TEST.SPLIT_VQA),
    bert_rationale_embeddings_path = cfg.BERT_EMBED_FILE % ('rationale', cfg.TEST.SPLIT_VQA))
num_vocab = data_reader.vocab_dict.num_vocab
num_answers = data_reader.num_combinations
module_names = data_reader.layout_dict.word_list
correct_label_batch_name = data_reader.correct_label_batch_name

# Eval files
if cfg.TEST.GEN_EVAL_FILE:
    eval_file = cfg.TEST.EVAL_FILE % (
        cfg.EXP_NAME, cfg.TEST.SPLIT_VQA, cfg.EXP_NAME, cfg.TEST.ITER)
    print('evaluation outputs will be saved to %s' % eval_file)
    os.makedirs(os.path.dirname(eval_file), exist_ok=True)
    output_qids_answers = []

dataset: tf.compat.v1.data.Dataset = data_reader.dataset
iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
next_element = iterator.get_next()

# Inputs and model
question_seq_batch = next_element['question_seq_batch']
correct_label_batch = next_element[correct_label_batch_name]
all_answers_seq_batch = next_element['all_answers_seq_batch']
all_answers_length_batch = next_element['all_answers_length_batch']
if data_reader.load_rationale:
    all_rationales_seq_batch = next_element['all_rationales_seq_batch']
    all_rationales_length_batch = next_element['all_rationales_length_batch']
else:
    all_rationales_seq_batch = None
    all_rationales_length_batch = None
if data_reader.load_bert:
    bert_question_embeddings_batch = next_element['bert_question_embeddings_batch']
    bert_answer_embeddings_batch = next_element['bert_answer_embeddings_batch']
    bert_rationale_embeddings_batch = next_element['bert_rationale_embeddings_batch']
else:
    bert_question_embeddings_batch = None
    bert_answer_embeddings_batch = None
    bert_rationale_embeddings_batch = None
question_length_batch = next_element['question_length_batch']
image_feat_batch = next_element['image_feat_batch']

model = Model(
    question_seq_batch,
    all_answers_seq_batch,
    all_rationales_seq_batch,
    question_length_batch,
    all_answers_length_batch,
    all_rationales_length_batch,
    bert_question_embeddings_batch,
    bert_answer_embeddings_batch,
    bert_rationale_embeddings_batch,
    image_feat_batch,
    num_vocab=num_vocab,
    num_choices=data_reader.num_combinations,
    module_names=module_names,
    is_training=False
)

# Load snapshot
if cfg.TEST.USE_EMA:
    ema = tf.train.ExponentialMovingAverage(decay=0.9)  # decay doesn't matter
    var_names = {
        (ema.average_name(v) if v in model.params else v.op.name): v
        for v in tf.global_variables()}
else:
    var_names = {v.op.name: v for v in tf.global_variables()}
snapshot_dir = cfg.TRAIN.SNAPSHOT_DIR % cfg.EXP_NAME
snapshot_file = os.path.join(snapshot_dir, f"{(cfg.TEST.ITER):08d}_{cfg.MODEL.VCR_TASK_TYPE}-{cfg.TEST.ITER}")
snapshot_saver = tf.train.Saver(var_names)
snapshot_saver.restore(sess, snapshot_file)

# Write results
result_dir = cfg.TEST.RESULT_DIR % (cfg.EXP_NAME, cfg.TEST.ITER)
vis_dir = os.path.join(
    result_dir, 'vqa_%s_%s' % (cfg.TEST.VIS_DIR_PREFIX, cfg.TEST.SPLIT_VQA))
os.makedirs(result_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)

sess.run(iterator.initializer)

# Run test
answer_correct, num_questions = 0, 0
try:
    for n_batch in itertools.count():
        if not data_reader.load_correct_answer:
            vqa_q_labels = -np.ones(
                len(data_reader.actual_batch_size), np.int32)
            if num_questions == 0:
                print('imdb has no answer labels. Using dummy labels.\n\n'
                    '**The final accuracy will be zero (no labels provided)**\n')
        
        fetch_list = [model.vqa_scores, next_element]
        answer_incorrect = num_questions - answer_correct

        if cfg.TEST.VIS_SEPARATE_CORRECTNESS:
            run_vis = (
                answer_correct < cfg.TEST.NUM_VIS_CORRECT or
                answer_incorrect < cfg.TEST.NUM_VIS_INCORRECT)
        else:
            run_vis = num_questions < cfg.TEST.NUM_VIS
        if run_vis:
            fetch_list.append(model.vis_outputs)

        fetch_list_val = sess.run(fetch_list)

        batch = fetch_list_val[1]

        # visualization
        if run_vis:
            model.vis_batch_vqa(
                data_reader, batch, fetch_list_val[-1], num_questions,
                answer_correct, answer_incorrect, vis_dir, num_answers)

        # compute accuracy

        vqa_scores_val = fetch_list_val[0]

        if not cfg.TRAIN.SOLVER.USE_SPARSE_SOFTMAX_LABELS:
            # Reshape the predictions into a softmax vector.
            vqa_scores_val = np.reshape(vqa_scores_val, (len(vqa_scores_val) // num_answers, num_answers))

        # Convert them into a one-hot encoded vector.
        vqa_predictions = np.argmax(vqa_scores_val, axis=1)

        if cfg.TEST.GEN_EVAL_FILE:
            samples = data_reader.imdb[batch['qid_list']]
            output_qids_answers += [
                { 'question_id': int(batch['qid_list'][i]), 'question_str': samples[i]['question_str'], 'answer': str(p), 'answer_str': ' '.join(samples[i]['all_answers'][batch['answer_index'][i]]) } for i, p in zip(range(0, len(batch['qid_list']), num_answers), vqa_predictions)
            ]

        if data_reader.load_correct_answer:
            vqa_labels = batch[data_reader.correct_label_batch_name]
            if not cfg.TRAIN.SOLVER.USE_SPARSE_SOFTMAX_LABELS:
            # Reshape the expected results into a one-hot encoded vector.
                vqa_labels = np.reshape(vqa_labels, (len(vqa_labels) // num_answers, num_answers))
                # Get the indices of the correct answer.
                vqa_labels = np.where(vqa_labels == 1.)[1]
        else:
            # dummy labels with all -1 (so accuracy will be zero)
            vqa_labels = -np.ones(vqa_scores_val.shape[0], np.int32)

        answer_correct += np.sum(vqa_predictions == vqa_labels)
        num_questions += len(vqa_labels)

        accuracy = answer_correct / num_questions

        if n_batch % 20 == 0:
            print(f'exp: {cfg.EXP_NAME}, iter = {cfg.TEST.ITER}, accumulated accuracy on {cfg.TEST.SPLIT_VQA} = {accuracy} ({answer_correct} / {answer_incorrect})')
except tf.errors.OutOfRangeError:
    print('Completed testing suite. Saving results.')

with open(eval_file, 'w') as f:
        json.dump(output_qids_answers, f, indent=2)
        print('prediction file written to ', eval_file)

with open(os.path.join(result_dir, f'vqa_results_{cfg.TEST.SPLIT_VQA}.txt'), 'w') as f:
    final_result = f'exp: {cfg.EXP_NAME}, iter = {cfg.TEST.ITER}, final accuracy on {cfg.TEST.SPLIT_VQA} = {accuracy} ({answer_correct} / {num_questions})'
    print(f'\n{final_result}')
    print(final_result, file=f)
