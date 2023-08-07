import sys
import os
from regex import regex as re
from typing import Callable

import tensorflow as tf

from models_vcr.model import Model
from models_vcr.config import build_cfg_from_argparse
from util.vcr_train.data_reader import DataReader

def create_data_reader(split: str, cfg):
    imdb_file = cfg.IMDB_FILE % split
    data_reader = DataReader(
        imdb_file, shuffle=True, one_pass=False, batch_size=cfg.TRAIN.BATCH_SIZE,
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
        bert_answer_embeddings_path = cfg.BERT_EMBED_FILE % ('answer', cfg.TRAIN.SPLIT_VQA),
        bert_rationale_embeddings_path = cfg.BERT_EMBED_FILE % ('rationale', cfg.TRAIN.SPLIT_VQA))

    return data_reader

def get_per_device_print_fn(tensor) -> Callable[[str], None]:
    device = tensor.device
    return lambda msg: print(f'{device}: {msg}')

def input_fn(is_training: bool = True):
    dataset: tf.compat.v1.data.Dataset = data_reader.init_dataset()
    if is_training:
        dataset = dataset.repeat(64)

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
    print_fn = get_per_device_print_fn(features['question_seq_batch'])
    
    print_fn('Initialising model')
    cfg = params['cfg']
    num_answers = params['num_answers']
    load_rationale = params['load_rationale']
    load_bert = params['load_bert']
    correct_label_batch_name = params['correct_label_batch_name']

    if labels is None:
        labels = features[correct_label_batch_name]

    model = Model(
        features['question_seq_batch'],
        features['all_answers_seq_batch'],
        features['all_rationales_seq_batch'] if load_rationale else None,
        features['question_length_batch'],
        features['all_answers_length_batch'],
        features['all_rationales_length_batch'] if load_rationale else None,
        features['bert_question_embeddings_batch'] if load_bert else None,
        features['bert_answer_embeddings_batch'] if load_bert else None,
        features['bert_rationale_embeddings_batch'] if load_bert and load_rationale else None,
        features['image_feat_batch'],
        num_vocab=params['num_vocab'],
        num_choices=params['num_combinations'],
        module_names=params['module_names'],
        is_training=mode==tf.estimator.ModeKeys.TRAIN,
        reuse=tf.AUTO_REUSE
    )

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        print_fn('Setting up loss and accuracy metrics')

        # Loss function
        if cfg.TRAIN.VQA_USE_SOFT_SCORE:
            soft_score_batch = tf.placeholder(tf.float32, [None], name='soft_score_batch')
            # Summing, instead of averaging over the choices
            loss_vqa = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=model.vqa_scores, labels=soft_score_batch), name='vqa_loss_function')
        elif cfg.TRAIN.SOLVER.USE_SPARSE_SOFTMAX_LABELS:
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
        loss_total = loss_train + cfg.TRAIN.WEIGHT_DECAY * model.l2_reg

        if not cfg.TRAIN.SOLVER.USE_SPARSE_SOFTMAX_LABELS:
            # Reshape the expected results into a one-hot encoded vector.
            vqa_q_labels = tf.reshape(labels, (tf.shape(labels)[0] // num_answers, num_answers))

            # Get the indices of the correct answer.
            vqa_q_labels = tf.where(tf.equal(vqa_q_labels, 1.))[:, 1]

            # Reshape the predictions into a softmax vector.
            vqa_scores_val = tf.reshape(model.vqa_scores, (tf.shape(model.vqa_scores)[0] // num_answers, num_answers))
        else:
            vqa_q_labels = labels
            vqa_scores_val = model.vqa_scores

        # Convert the logits into the predicted indices of the correct answer.
        vqa_predictions = tf.argmax(vqa_scores_val, axis=1, output_type=tf.int32)
        
        vqa_q_labels = tf.cast(vqa_q_labels, tf.int64)
        vqa_predictions = tf.cast(vqa_predictions, tf.int64)

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

        grads_and_vars = solver.compute_gradients(loss_total)
        global_step = tf.train.get_or_create_global_step()

        if cfg.TRAIN.CLIP_GRADIENTS:
            print_fn(f'clipping gradients to max norm: {cfg.TRAIN.GRAD_MAX_NORM:f}')
            gradients, variables = zip(*grads_and_vars)
            gradients, _ = tf.clip_by_global_norm(gradients, cfg.TRAIN.GRAD_MAX_NORM)
            grads_and_vars = zip(gradients, variables)
        solver_op = solver.apply_gradients(grads_and_vars, global_step=global_step)

        print_fn('Initializing exponential moving average.')
        with tf.control_dependencies([solver_op]):
            ema._num_updates = global_step
            ema_op = ema.apply(model.params)
            train_op = tf.group(ema_op)

        # Create a hook to update the metrics per run
        class MetricHook(tf.train.SessionRunHook):
            def before_run(self, run_context):
                return tf.train.SessionRunArgs([accuracy[1]])

        metric_hook = MetricHook()
        with tf.device('/CPU:0'):
            tf.summary.scalar('accuracy', accuracy[0])
            summary_hook = tf.estimator.SummarySaverHook(save_steps=cfg.TRAIN.LOG_INTERVAL, summary_op=tf.summary.merge_all())
        # Print the accuracy to stdout every LOG_INTERVAL steps.
        logging_hook = tf.train.LoggingTensorHook({ 'accuracy': accuracy[0] }, every_n_iter=cfg.TRAIN.LOG_INTERVAL)

        print_fn('Training mode initialised')
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss_total, train_op=train_op, training_hooks=[ summary_hook, metric_hook, logging_hook ])

    elif mode == tf.estimator.ModeKeys.EVAL:
        # Calculate additional useful metrics for eval.
        precision = tf.metrics.precision(labels=vqa_q_labels, predictions=vqa_predictions, name='precision_op')
        recall = tf.metrics.recall(labels=vqa_q_labels, predictions=vqa_predictions, name='recall_op')
        eval_metric_ops = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }

        # Log the metrics to tensorboard
        with tf.device('/CPU:0'):
            tf.summary.scalar('accuracy', accuracy[1])
            tf.summary.scalar('precision', precision[1])
            tf.summary.scalar('recall', recall[1])

        print_fn('Eval mode initialised.')

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss_total, eval_metric_ops=eval_metric_ops)
    else:
        print_fn('Prediction mode initialised.')
        return tf.estimator.EstimatorSpec(mode=mode, predictions=model.vqa_scores)

# Load config
cfg = build_cfg_from_argparse()

# Start session
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cfg.GPU_ID)
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

# Enables support for XLA optimisation
tf.config.optimizer.set_jit('autoclustering')

# Data files
data_reader = create_data_reader(cfg.TRAIN.SPLIT_VQA, cfg)

# Due to the variables tf.train.ExponentialMovingAverage creates internally, it can't be declared in the model_fn without conflicts.
# Instead, it must be declared outside the model_fn to avoid the conflicts between it and the MirroredStrategy variables.
ema = tf.train.ExponentialMovingAverage(decay=cfg.TRAIN.EMA_DECAY)

# Multi-GPU configuration
strategy = tf.contrib.distribute.MirroredStrategy()

snapshot_dir = cfg.TRAIN.SNAPSHOT_DIR % cfg.EXP_NAME
sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=cfg.GPU_MEM_GROWTH))
config = tf.estimator.RunConfig(
    model_dir=snapshot_dir,
    train_distribute=strategy,
    eval_distribute=strategy,
    session_config=sess_config,
    log_step_count_steps=cfg.TRAIN.LOG_INTERVAL,
    save_summary_steps=cfg.TRAIN.LOG_INTERVAL,
    save_checkpoints_steps=100, #cfg.TRAIN.SNAPSHOT_INTERVAL,
    keep_checkpoint_max=0
)

# Initialise memory profiler for additional debugging.
profiler_hook = tf.train.ProfilerHook(save_steps=cfg.TRAIN.SNAPSHOT_INTERVAL, output_dir=snapshot_dir, show_dataflow=True, show_memory=True)

print('Main: Initializing Estimator.')
model = tf.estimator.Estimator(model_fn=model_fn, config=config, params=get_model_fn_params(cfg, data_reader))
print('Main: Initialised.')
print('Main: Beginning training.')
model.train(input_fn=lambda: input_fn(is_training=True), steps=cfg.TRAIN.MAX_ITER, hooks=[ profiler_hook ])
print('Main: Training completed. Evaluating checkpoints.')
checkpoints = [ os.path.join(snapshot_dir, c.group(1)) for c in [re.match(pattern = '(model.ckpt-[^0].*).data-00000.*', string=s) for s in os.listdir(snapshot_dir)] if c is not None]
if checkpoints is None or len(checkpoints) == 0:
    print('Main: No checkpoints to evaluate. Exiting.')
    sys.exit(0)

print('Main: Creating eval data_reader.')
del data_reader, model
data_reader = create_data_reader(cfg.EVAL.SPLIT_VQA, cfg)
print('Main: Creating new Estimator with updated eval data_reader')
model = tf.estimator.Estimator(model_fn=model_fn, config=config, params=get_model_fn_params(cfg, data_reader))

for checkpoint in checkpoints:
    print(f'Main: Evaluating checkpoint {checkpoint}')
    eval_metrics = model.evaluate(
        input_fn=lambda: input_fn(is_training=False),
        steps=cfg.EVAL.MAX_ITER if cfg.EVAL.MAX_ITER > 0 else None,
        checkpoint_path=checkpoint
    )
    print(f'Main: Evaluation results for checkpoint {checkpoint}: {eval_metrics}')
print('Main: Completed evaluation. Exiting.')
