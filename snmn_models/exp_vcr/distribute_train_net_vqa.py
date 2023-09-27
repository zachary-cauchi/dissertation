import sys
import os
import csv
import math
import json
import numpy as np
from regex import regex as re
from typing import Callable

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit' # --tf_xla_auto_jit=2
# os.environ['XLA_FLAGS'] = '--xla_hlo_profile'

import tensorflow as tf
from tensorflow.contrib.distribute import MirroredStrategy

from models_vcr.model import Model
from models_vcr.config import build_cfg_from_argparse
from util.vcr_train.data_reader import DataReader
from util.text_processing import VocabDict

def get_f1_score(labels, predictions):
    precision_val, precision_op = tf.metrics.precision(labels=labels, predictions=predictions, name='precision_op')
    recall_val, recall_op = tf.metrics.recall(labels=labels, predictions=predictions, name='recall_op')
    
    # Avoid zero division by adding a small constant
    epsilon = 1e-7
    
    # Calculate F1 score
    f1_val = 2. * (precision_val * recall_val) / (precision_val + recall_val + epsilon)
    with tf.control_dependencies([precision_op, recall_op]):
        f1_op = tf.identity(f1_val, name="f1_score")
    
    return f1_val, f1_op

def create_data_reader(split: str, cfg):
    imdb_file = cfg.IMDB_FILE % split
    data_reader = DataReader(
        imdb_file, shuffle=True, one_pass=False, batch_size=cfg.TRAIN.BATCH_SIZE,
        vocab_question_file=cfg.VOCAB_QUESTION_FILE,
        T_q_encoder=cfg.MODEL.T_Q_ENCODER,
        T_a_encoder=cfg.MODEL.T_A_ENCODER,
        T_r_encoder=cfg.MODEL.T_R_ENCODER,
        vocab_answer_file=cfg.VOCAB_ANSWER_FILE % split,
        load_gt_layout=cfg.TRAIN.USE_GT_LAYOUT,
        vocab_layout_file=cfg.VOCAB_LAYOUT_FILE, T_decoder=cfg.MODEL.T_CTRL,
        load_soft_score=cfg.TRAIN.VQA_USE_SOFT_SCORE,
        feed_answers_with_input=cfg.MODEL.INPUT.USE_ANSWERS,
        vcr_task_type=cfg.MODEL.VCR_TASK_TYPE,
        use_sparse_softmax_labels=cfg.TRAIN.SOLVER.USE_SPARSE_SOFTMAX_LABELS,
        load_bert_embeddings = cfg.USE_BERT_SENTENCE_EMBED,
        bert_embeddings_path = os.path.join(cfg.BERT_EMBED_DIR, split),
        external_true_answers_file = cfg.TRAIN.EXTERNAL_TRUE_ANSWERS_FILE if cfg.TRAIN.USE_EXTERNAL_TRUE_ANSWERS_FILE else '')

    return data_reader

def get_per_device_print_fn(tensor) -> Callable[[str], None]:
    device = tensor.device
    return lambda msg: print(f'{device}: {msg}')

def input_fn(is_training: bool = True):
    if is_training:
        repeat_count = (math.ceil(data_reader.imdb_count / cfg.TRAIN.MAX_ITER) + 1) * data_reader.actual_batch_size
        dataset: tf.compat.v1.data.Dataset = data_reader.init_dataset().repeat(repeat_count)
    else:
        dataset: tf.compat.v1.data.Dataset = data_reader.init_dataset()

    # dataset.apply(self=dataset, transformation_func=tf.data.experimental.prefetch_to_device())

    return dataset

def get_model_fn_params(cfg, data_reader: DataReader):
    return {
        'cfg': cfg,
        'num_vocab': data_reader.vocab_dict.num_vocab,
        'num_answers': data_reader.num_answers,
        'num_combinations': data_reader.num_combinations,
        'load_rationale': data_reader.load_rationale,
        'load_bert': data_reader.load_bert,
        'module_names': data_reader.layout_dict.word_list,
        'correct_label_batch_name': data_reader.correct_label_batch_name
    }

def model_fn(features, labels, mode: tf.estimator.ModeKeys, params):
    print_fn = get_per_device_print_fn(features['question_id'])
    
    print_fn('Initialising model')
    cfg = params['cfg']
    num_answers = params['num_answers']
    num_combinations = params['num_combinations']
    load_rationale = params['load_rationale']
    load_bert = params['load_bert']
    correct_label_batch_name = params['correct_label_batch_name']

    # if labels is None:
    #     labels = features[correct_label_batch_name]

    model = Model(
        features['question_sequence'],
        features['all_answers_sequences'],
        features['all_rationales_sequences'] if load_rationale else None,
        features['question_length'],
        features['all_answers_length'],
        features['all_rationales_length'] if load_rationale else None,
        features['bert_question_embedding'] if load_bert else None,
        features['bert_answer_embedding'] if load_bert else None,
        features['bert_rationale_embedding'] if load_bert and load_rationale else None,
        features['image_feat'],
        num_vocab=params['num_vocab'],
        num_choices=num_combinations,
        module_names=params['module_names'],
        is_training=mode==tf.estimator.ModeKeys.TRAIN,
        reuse=tf.AUTO_REUSE,
        use_cudnn_lstm=cfg.MODEL.INPUT.USE_CUDNN_LSTM,
        use_shared_lstm=cfg.MODEL.INPUT.USE_SHARED_LSTM
    )

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        print_fn('Setting up loss and accuracy metrics')
        global_step = tf.compat.v1.train.get_or_create_global_step()

        # Loss function
        if cfg.TRAIN.SOLVER.USE_SPARSE_SOFTMAX_LABELS:
            loss_vqa = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=model.vqa_scores, labels=tf.stop_gradient(labels)), name='vqa_sparse_softmax_loss_function')
        else:
            loss_vqa = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=model.vqa_scores, labels=tf.stop_gradient(labels)), name='vqa_sigmoid_loss_function')

        # Loss function for expert layout.
        if cfg.TRAIN.USE_GT_LAYOUT:
            gt_layout_question_batch = tf.placeholder(tf.int32, [None, None], name='gt_layout_question_batch')
            loss_layout = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=model.module_logits, labels=gt_layout_question_batch))
        else:
            loss_layout = tf.convert_to_tensor(0.)
        loss_rec = model.rec_loss
        loss_train = (loss_vqa * cfg.TRAIN.VQA_LOSS_WEIGHT +
                    loss_layout * cfg.TRAIN.LAYOUT_LOSS_WEIGHT +
                    loss_rec * cfg.TRAIN.REC_LOSS_WEIGHT)
        loss_total = loss_train + cfg.TRAIN.WEIGHT_DECAY * model.elastic_net_reg

        if not cfg.TRAIN.SOLVER.USE_SPARSE_SOFTMAX_LABELS:
            # Reshape the expected results into a one-hot encoded vector.
            vqa_q_labels = tf.reshape(labels, [-1, num_combinations])

            # Get the indices of the correct answer.
            vqa_q_labels = tf.where(tf.equal(vqa_q_labels, 1.))[:, 1]

            # Reshape the predictions into a softmax vector.
            vqa_scores_val = tf.reshape(model.vqa_scores, [-1, num_combinations])
        else:
            vqa_q_labels = labels
            vqa_scores_val = model.vqa_scores

        # Convert the logits into the predicted indices of the correct answer.
        vqa_predictions = tf.argmax(vqa_scores_val, axis=1, output_type=tf.int32)
        vqa_q_labels = tf.cast(vqa_q_labels, tf.int32)

        # tf.debugging.assert_shapes([(vqa_q_labels, (None,)), (vqa_predictions, (None,))])
        accuracy = tf.metrics.accuracy(labels=vqa_q_labels, predictions=vqa_predictions, name='accuracy_op')

    if mode == tf.estimator.ModeKeys.TRAIN:
        print_fn('Setting up optimizer.')
        # Train with Adam
        solver = tf.train.AdamOptimizer(learning_rate=cfg.TRAIN.SOLVER.LR)

        # This will do a few things:
        #   * Enable Automatic Mixed Precision (AMP) which will mix in float16 types in the graph.
        #   * Enable XLA graph optimisation which compiles the generated graph for improved performance.
        #   * Enables use of Tensor Cores on supporting Nvidia GPUs.
        solver = tf.train.experimental.enable_mixed_precision_graph_rewrite(solver, loss_scale='dynamic')

        # TODO: Consider dynamic loss scale such as tf.train.experiential.DynamicLossScale?
        grads_and_vars = solver.compute_gradients(loss_total)

        if cfg.TRAIN.CLIP_GRADIENTS:
            print_fn(f'clipping gradients to max norm: {cfg.TRAIN.GRAD_MAX_NORM:f}')
            gradients, variables = zip(*grads_and_vars)
            nan_check_unclipped_gradients = [ tf.debugging.assert_all_finite(gradient, message=f'NaN detected in UNCLIPPED gradient "{gradient.name}"') for gradient in gradients]
            gradients, _ = tf.clip_by_global_norm(gradients, cfg.TRAIN.GRAD_MAX_NORM, name='perform_gradient_clipping')
            grads_and_vars = zip(gradients, variables)
        solver_op = solver.apply_gradients(grads_and_vars, global_step=global_step)

        print_fn('Initializing exponential moving average.')
        with tf.control_dependencies([solver_op]):
            ema._num_updates = global_step
            ema_op = ema.apply(model.params)
            train_op = tf.group(ema_op, name='ema_train_op')

        # TODO: NaN values appearing in gradients for output_unit. Investigate further.
        nan_check_loss = tf.debugging.assert_all_finite(loss_total, message='NaN detected in loss')
        nan_check_reg = tf.debugging.assert_all_finite(model.elastic_net_reg, message='NaN detected in regularization')
        nan_check_clipped_gradients = [ tf.debugging.assert_all_finite(gradient, message=f'NaN detected in CLIPPED gradient "{gradient.name}"') for gradient in gradients]
        nan_check_params = [ tf.debugging.assert_all_finite(v, message=f'NaN detected in model param "{v.name}"') for v in model.params]

        nan_checks = tf.group(nan_check_loss, nan_check_reg, *nan_check_unclipped_gradients, *nan_check_clipped_gradients, *nan_check_params, name='all_assertions')

        # Create a hook to update the metrics per run
        class MetricHook(tf.train.SessionRunHook):
            def before_run(self, run_context):
                return tf.estimator.SessionRunArgs([accuracy[1]])

        metric_hook = MetricHook()
        with tf.device('/CPU:0'):
            tf.summary.scalar('accuracy', accuracy[0])
            tf.summary.scalar('loss/loss-vqa', loss_vqa)
            tf.summary.scalar('reg/l1-reg', model.l1_reg)
            tf.summary.scalar('reg/l2-reg', model.l2_reg)
            tf.summary.scalar('reg/elastic-net-reg', model.elastic_net_reg)
            summary_hook = tf.estimator.SummarySaverHook(save_steps=cfg.TRAIN.LOG_INTERVAL, summary_op=tf.summary.merge_all())
        # Print the accuracy to stdout every LOG_INTERVAL steps.
        logging_hook = tf.train.LoggingTensorHook({ 'accuracy': accuracy[0], 'loss-vqa': loss_vqa }, every_n_iter=cfg.TRAIN.LOG_INTERVAL)

        print_fn('Training mode initialised')
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss_total, train_op=train_op, training_hooks=[ summary_hook, metric_hook, logging_hook ])

    elif mode == tf.estimator.ModeKeys.EVAL:
        # Calculate additional useful metrics for eval.
        precision = tf.metrics.precision(labels=vqa_q_labels, predictions=vqa_predictions, name='precision_op')
        recall = tf.metrics.recall(labels=vqa_q_labels, predictions=vqa_predictions, name='recall_op')
        f1_score = get_f1_score(labels=vqa_q_labels, predictions=vqa_predictions)

        eval_metric_ops = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1-score': f1_score
        }

        # Log the metrics to tensorboard
        with tf.device('/CPU:0'):
            tf.summary.scalar('accuracy', accuracy[1])
            tf.summary.scalar('precision', precision[1])
            tf.summary.scalar('recall', recall[1])
            tf.summary.scalar('f1-score', f1_score[1])

        print_fn('Eval mode initialised.')

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss_total, eval_metric_ops=eval_metric_ops)

    elif mode == tf.estimator.ModeKeys.PREDICT:

        if not cfg.TRAIN.SOLVER.USE_SPARSE_SOFTMAX_LABELS:
            # Reshape the predictions into a softmax vector.
            vqa_scores_val = tf.reshape(model.vqa_scores, [-1, num_combinations])
        else:
            vqa_scores_val = model.vqa_scores

        # Convert the logits into the predicted indices of the correct answer.
        vqa_predictions = tf.argmax(vqa_scores_val, axis=1, output_type=tf.int32)

        answer_tokens = tf.transpose(features['all_answers_sequences'])
        if data_reader.vcr_task_type == 'Q_2_A':
            answer = vqa_predictions
        elif data_reader.vcr_task_type == 'QA_2_R' and data_reader.load_correct_answer:
            answer = features['valid_answers_index']
        else:
            answer = data_reader.i_ans_range[vqa_predictions]

        predictions = {
            'logits': tf.expand_dims(vqa_scores_val, axis=0),
            'question_id': tf.expand_dims(features['question_id'][::4], axis=0),
            'question_tokens': tf.expand_dims(features['question_tokens'][::4], axis=0),
            'answer': tf.expand_dims(answer, axis=0),
            'answer_tokens': tf.expand_dims(answer_tokens, axis=0),
        }

        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    else:
        print_fn('Prediction mode initialised.')
        return tf.estimator.EstimatorSpec(mode=mode, predictions=model.vqa_scores)

def get_checkpoints(dir: str):
    checkpoints = [ os.path.join(dir, c.group(1)) for c in [re.match(pattern = r'(model.ckpt-[^0].*).data-00000.*', string=s) for s in os.listdir(dir)] if c is not None]
    return sorted(checkpoints, key = lambda c: int(re.search(r'-(\d+)$', c).group(0)), reverse=True)

# Load config
cfg = build_cfg_from_argparse()
snapshot_dir = cfg.TRAIN.SNAPSHOT_DIR % cfg.EXP_NAME

if 'CUDA_VISIBLE_DEVICES' not in os.environ and cfg.MAX_GPUS < 2:
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cfg.GPU_ID)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

# Enables support for XLA optimisation
# os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_LOG_PATH'] = os.path.join(snapshot_dir, 'xla')
tf.config.optimizer.set_jit('autoclustering')

# Data files
data_reader = create_data_reader(cfg.TRAIN.SPLIT_VQA, cfg)

# Due to the variables tf.train.ExponentialMovingAverage creates internally, it can't be declared in the model_fn without conflicts.
# Instead, it must be declared outside the model_fn to avoid the conflicts between it and the MirroredStrategy variables.
ema = tf.train.ExponentialMovingAverage(decay=cfg.TRAIN.EMA_DECAY)

# Multi-GPU configuration
train_strategy = MirroredStrategy(num_gpus=cfg.MAX_GPUS)
eval_strategy = MirroredStrategy(num_gpus=1)

sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=cfg.GPU_MEM_GROWTH))
# If using the bidirectional_dynamic_rnn, we have to enable this due to an error with how it asserts the embedding sizes.
if not cfg.MODEL.INPUT.USE_CUDNN_LSTM:
    sess_config.allow_soft_placement = True

config = tf.estimator.RunConfig(
    model_dir=snapshot_dir,
    train_distribute=train_strategy,
    eval_distribute=eval_strategy,
    session_config=sess_config,
    log_step_count_steps=cfg.TRAIN.LOG_INTERVAL,
    save_summary_steps=cfg.TRAIN.LOG_INTERVAL,
    save_checkpoints_steps=cfg.TRAIN.SNAPSHOT_INTERVAL,
    keep_checkpoint_max=0
)

# Initialise memory profiler for additional debugging.
profiler_hook = tf.train.ProfilerHook(save_steps=cfg.TRAIN.SNAPSHOT_INTERVAL, output_dir=snapshot_dir, show_dataflow=True, show_memory=True)

if cfg.TRAIN.START_ITER > 0:
    start_checkpoint = os.path.join(snapshot_dir, f'model.ckpt-{str(cfg.TRAIN.START_ITER)}')
else:
    start_checkpoint = None

# Perform training

if cfg.RUN.TRAIN:
    print('Main: Initializing training Estimator.')
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=config, params=get_model_fn_params(cfg, data_reader), warm_start_from=start_checkpoint)
    print('Main: Initialised.')
    print('Main: Beginning training.')
    estimator.train(input_fn=lambda: input_fn(is_training=True), steps=cfg.TRAIN.MAX_ITER, hooks=[ profiler_hook ])
    print('Main: Training completed.')
    del data_reader, estimator

# Perform eval
checkpoints = get_checkpoints(snapshot_dir)
best_checkpoint = None

if cfg.RUN.EVAL:
    if checkpoints is None or len(checkpoints) == 0:
        print('Main: No checkpoints to evaluate.')
        sys.exit(0)
    print('Main: Creating eval data_reader.')
    data_reader = create_data_reader(cfg.EVAL.SPLIT_VQA, cfg)
    print('Main: Creating new Estimator with eval data_reader')
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=config, params=get_model_fn_params(cfg, data_reader))

    # Eval each checkpoint and find best checkpoint by accuracy.

    highest_acc = 0.
    for checkpoint in checkpoints:
        print(f'Main: Evaluating checkpoint {checkpoint}')
        eval_metrics = estimator.evaluate(
            input_fn=lambda: input_fn(is_training=False),
            checkpoint_path=checkpoint
        )
        print(f'Main: Evaluation results for checkpoint {checkpoint}: {eval_metrics}')

        if eval_metrics['accuracy'] > highest_acc:
            highest_acc = eval_metrics['accuracy']
            best_checkpoint = checkpoint

    print('Main: Completed evaluation.')
    print(f'Main: Best checkpoint - with an accuracy of {highest_acc}% is {checkpoint}')

test_checkpoint = best_checkpoint if best_checkpoint is not None else os.path.join(snapshot_dir, cfg.TEST.CHECKPOINT)

if cfg.RUN.TEST:
    data_reader = create_data_reader(cfg.TEST.SPLIT_VQA, cfg)
    print('Main: Creating new Estimator with test data_reader')
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=config, params=get_model_fn_params(cfg, data_reader))

    print(f'Main: Testing checkpoint {test_checkpoint}')

    if cfg.TEST.GEN_EVAL_FILE:
        logits_eval_file = f'./exp_vcr/eval_outputs/{cfg.EXP_NAME}/vcr_{os.path.basename(test_checkpoint)}_leaderboard.csv'
        results_eval_file = f'./exp_vcr/eval_outputs/{cfg.EXP_NAME}/vcr_{os.path.basename(test_checkpoint)}_results.json'
        
        print(f'Main: Prediction outputs will be saved to {results_eval_file}')
        print(f'Main: Prediction logits will be saved in {logits_eval_file}')

        os.makedirs(os.path.dirname(logits_eval_file), exist_ok=True)
        output_qids_answers = []

    preds = {
        'logits': [],
        'question_id': [],
        'annot_id': [],
        'question_tokens': [],
        'answer': [],
        'answer_tokens': [],
    }

    vocab_dict = VocabDict(cfg.VOCAB_QUESTION_FILE, first_token_only=True)

    for pred in estimator.predict(input_fn=lambda: input_fn(is_training=False), checkpoint_path=test_checkpoint):
        preds['logits'].extend([ list([ float(logit) for logit in logits ]) for logits in pred['logits'] ])
        preds['question_id'].extend([ int(id) for id in pred['question_id'] ])
        preds['annot_id'].extend([ f'{cfg.TEST.SPLIT_VQA}-{id}' for id in pred['question_id'] ])
        preds['question_tokens'].extend([ [ b.decode() for b in tokens if b != b'' ] for tokens in pred['question_tokens'] ])
        preds['answer'].extend([ int(a) for a in pred['answer'] ])
        answer_tokens_reshaped = np.reshape(pred['answer_tokens'], [pred['answer'].shape[0], data_reader.num_combinations, data_reader.T_a_encoder])
        preds['answer_tokens'].extend([ list([ list([ vocab_dict.idx2word(token) for token in tokens if token != 0 ]) for tokens in answers ]) for answers in answer_tokens_reshaped ])

    del answer_tokens_reshaped

    with open(logits_eval_file, 'w') as f:
        a0 = []
        a1 = []
        a2 = []
        a3 = []
        for l in preds['logits']:
            a0.append(l[0])
            a1.append(l[1])
            a2.append(l[2])
            a3.append(l[3])

        writer = csv.writer(f)
        writer.writerow(['annot_id', 'answer_0', 'answer_1', 'answer_2', 'answer_3'])
        for i in range(len(preds['logits'])):
            writer.writerow([preds['annot_id'][i], a0[i], a1[i], a2[i], a3[i]])

    print('Main: Logits file saved')

    with open(results_eval_file, 'w') as f:
        json_data = []
        for i in range(len(preds['logits'])):
            json_data.append({
                'question_id': preds['question_id'][i],
                'annot_id': preds['annot_id'][i],
                'question_tokens': preds['question_tokens'][i],
                'answer': preds['answer'][i],
                'answer_tokens': preds['answer_tokens'][i],
                'logits': preds['logits'][i],
            })

        json_data = sorted(json_data, key = lambda x: x['question_id'])

        json.dump(json_data, f, indent=2)

    print('Main: Results file saved')

print('Main: Exiting')
