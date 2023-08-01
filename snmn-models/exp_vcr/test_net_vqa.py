import argparse
import os
import json
import numpy as np
import tensorflow as tf

from models_vcr.model import Model
from models_vcr.config import build_cfg_from_argparse
from util.vcr_train.data_reader import DataReader

# Load config
cfg = build_cfg_from_argparse()

# Start session
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
num_vocab = data_reader.batch_loader.vocab_dict.num_vocab
num_answers = data_reader.batch_loader.num_combinations
module_names = data_reader.batch_loader.layout_dict.word_list

# Eval files
if cfg.TEST.GEN_EVAL_FILE:
    eval_file = cfg.TEST.EVAL_FILE % (
        cfg.EXP_NAME, cfg.TEST.SPLIT_VQA, cfg.EXP_NAME, cfg.TEST.ITER)
    print('evaluation outputs will be saved to %s' % eval_file)
    os.makedirs(os.path.dirname(eval_file), exist_ok=True)
    output_qids_answers = []

if data_reader.data_params['vcr_task_type'] == 'Q_2_A':
    correct_label_batch_name = 'answer_label_batch'
elif data_reader.data_params['vcr_task_type'] == 'QA_2_R':
    correct_label_batch_name = 'rationale_label_batch'
elif data_reader.data_params['vcr_task_type'] == 'Q_2_AR':
    correct_label_batch_name = 'answer_and_rationale_label_batch'

# Inputs and model
question_seq_batch = tf.placeholder(tf.int32, [None, None], name='question_seq_batch')
correct_label_batch = tf.placeholder(tf.int32, [None], name=f'correct_{correct_label_batch_name}')
all_answers_seq_batch = tf.placeholder(tf.int32, [None, None], name='all_answers_seq_batch')
all_answers_length_batch = tf.placeholder(tf.int32, [None], name='all_answers_length_batch')
if data_reader.batch_loader.load_rationale:
    rationale_label_batch = tf.placeholder(tf.float32, [None], name='rationale_label_batch')
    all_rationales_seq_batch = tf.placeholder(tf.int32, [None, None], name='all_rationales_seq_batch')
    all_rationales_length_batch = tf.placeholder(tf.int32, [None], name='all_rationales_length_batch')
else:
    rationale_label_batch = None
    all_rationales_seq_batch = None
    all_rationales_length_batch = None
if data_reader.batch_loader.load_bert:
    bert_question_embeddings_batch = tf.placeholder(tf.float16, [None, None, data_reader.batch_loader.bert_dim], name='bert_question_embeddings_batch')
    bert_answer_embeddings_batch = tf.placeholder(tf.float16, [None, None, data_reader.batch_loader.bert_dim], name='bert_answer_embeddings_batch')
    bert_rationale_embeddings_batch = tf.placeholder(tf.float16, [None, None, data_reader.batch_loader.bert_dim], name='bert_rationale_embeddings_batch')
else:
    bert_question_embeddings_batch = None
    bert_answer_embeddings_batch = None
    bert_rationale_embeddings_batch = None
question_length_batch = tf.placeholder(tf.int32, [None], name='question_length_batch')
image_feat_batch = tf.placeholder(
    tf.float32, [None, cfg.MODEL.H_FEAT, cfg.MODEL.W_FEAT, cfg.MODEL.FEAT_DIM], name='image_feat_batch')
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
    num_choices=data_reader.batch_loader.num_combinations,
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
# snapshot_file = cfg.TEST.SNAPSHOT_FILE % (cfg.EXP_NAME, cfg.TEST.ITER)
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

# Run test
answer_correct, num_questions = 0, 0
for n_batch, batch in enumerate(data_reader.batches()):
    if correct_label_batch_name not in batch:
        batch[correct_label_batch_name] = -np.ones(
            len(batch['image_feat_batch']), np.int32)
        if num_questions == 0:
            print('imdb has no answer labels. Using dummy labels.\n\n'
                  '**The final accuracy will be zero (no labels provided)**\n')
    
    fetch_list = [model.vqa_scores]
    answer_incorrect = num_questions - answer_correct

    if cfg.TEST.VIS_SEPARATE_CORRECTNESS:
        run_vis = (
            answer_correct < cfg.TEST.NUM_VIS_CORRECT or
            answer_incorrect < cfg.TEST.NUM_VIS_INCORRECT)
    else:
        run_vis = num_questions < cfg.TEST.NUM_VIS
    if run_vis:
        fetch_list.append(model.vis_outputs)

    feed_dict = {
        question_seq_batch: batch['question_seq_batch'],
        question_length_batch: batch['question_length_batch'],
        image_feat_batch: batch['image_feat_batch'],
        all_answers_seq_batch: batch['all_answers_seq_batch'],
        all_answers_length_batch: batch['all_answers_length_batch']
    }

    if data_reader.batch_loader.load_rationale:
        feed_dict.update(
            all_rationales_seq_batch=batch['all_rationales_seq_batch'],
            all_rationales_length_batch=batch['all_rationales_length_batch']
        )

    if data_reader.batch_loader.load_bert:
        feed_dict[bert_question_embeddings_batch] = batch['bert_question_embeddings_batch']
        feed_dict[bert_answer_embeddings_batch] = batch['bert_answer_embeddings_batch']
        if data_reader.batch_loader.load_rationale:
            feed_dict[bert_rationale_embeddings_batch] = batch['bert_rationale_embeddings_batch']

    fetch_list_val = sess.run(fetch_list, feed_dict=feed_dict)

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
        qid_list = [batch['qid_list'][i] for i in range(0, len(batch['qid_list']) // num_answers, num_answers)]
        output_qids_answers += [
            {'question_id': int(qid), 'answer': p.item(), 'answer_str': ' '.join(batch['all_answers_list'][(i * num_answers) + p])}
            for i, (qid, p) in enumerate(zip(qid_list, vqa_predictions))]

    if data_reader.batch_loader.load_correct_answer:
        vqa_labels = batch['answer_label_batch']
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

with open(eval_file, 'w') as f:
        json.dump(output_qids_answers, f, indent=2)
        print('prediction file written to ', eval_file)

with open(os.path.join(result_dir, f'vqa_results_{cfg.TEST.SPLIT_VQA}.txt'), 'w') as f:
    final_result = f'exp: {cfg.EXP_NAME}, iter = {cfg.TEST.ITER}, final accuracy on {cfg.TEST.SPLIT_VQA} = {accuracy} ({answer_correct} / {num_questions})'
    print(f'\n{final_result}')
    print(final_result, file=f)
