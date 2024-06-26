import math
import os
import numpy as np
import tensorflow as tf
import time
import signal
import sys

from models_vcr.model import Model
from models_vcr.config import build_cfg_from_argparse
from util.vcr_train.data_reader import DataReader

# Load config
cfg = build_cfg_from_argparse()

# Start session
if not 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPU_ID)
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

# TODO: Enable XLA graph optimisations
# tf.config.optimizer.set_jit('autoclustering')

sess = tf.Session(config=tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=cfg.GPU_MEM_GROWTH)))

batch_size=cfg.TRAIN.BATCH_SIZE

# Data files
imdb_file = cfg.IMDB_FILE % cfg.TRAIN.SPLIT_VQA
data_reader = DataReader(
    imdb_file, shuffle=True, one_pass=False, batch_size=batch_size,
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
num_vocab = data_reader.vocab_dict.num_vocab
num_answers = data_reader.num_combinations
module_names = data_reader.layout_dict.word_list
correct_label_batch_name = data_reader.correct_label_batch_name

repeat_count = (math.ceil(data_reader.imdb_count / cfg.TRAIN.MAX_ITER) + 1) * data_reader.actual_batch_size
dataset: tf.compat.v1.data.Dataset = data_reader.init_dataset().repeat(repeat_count)
iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
next_element = iterator.get_next()

question_seq_batch = next_element[0]['question_sequence']
correct_label_batch = next_element[1]
all_answers_seq_batch = next_element[0]['all_answers_sequences']
all_answers_length_batch = next_element[0]['all_answers_length']
if data_reader.load_rationale:
    all_rationales_seq_batch = next_element[0]['all_rationales_sequences']
    all_rationales_length_batch = next_element[0]['all_rationales_length']
else:
    all_rationales_seq_batch = None
    all_rationales_length_batch = None
if data_reader.load_bert:
    bert_question_embeddings_batch = next_element[0]['bert_question_embeddings_batch']
    bert_answer_embeddings_batch = next_element[0]['bert_answer_embeddings_batch']
    if data_reader.load_rationale:
        bert_rationale_embeddings_batch = next_element[0]['bert_rationale_embeddings_batch']
    else:
        bert_rationale_embeddings_batch = None
else:
    bert_question_embeddings_batch = None
    bert_answer_embeddings_batch = None
    bert_rationale_embeddings_batch = None
question_length_batch = next_element[0]['question_length']
image_feat_batch = next_element[0]['image_feat']

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
    is_training=True,
    use_cudnn_lstm=cfg.MODEL.INPUT.USE_CUDNN_LSTM
)

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
            logits=model.vqa_scores, labels=tf.stop_gradient(correct_label_batch)), name='vqa_sparse_softmax_loss_function')
else:
    loss_vqa = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=model.vqa_scores, labels=tf.stop_gradient(correct_label_batch)), name='vqa_sigmoid_loss_function')

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

# Train with Adam
solver = tf.train.AdamOptimizer(learning_rate=cfg.TRAIN.SOLVER.LR)
# This will do a few things:
#   * Enable Automatic Mixed Precision (AMP) which will mix in float16 types in the graph.
#   * Enable XLA graph optimisation which compiles the generated graph for improved performance.
#   * Enables use of Tensor Cores on supporting Nvidia GPUs.
# solver = tf.train.experimental.enable_mixed_precision_graph_rewrite(solver, loss_scale='dynamic')
grads_and_vars = solver.compute_gradients(loss_total)
if cfg.TRAIN.CLIP_GRADIENTS:
    print('clipping gradients to max norm: %f' % cfg.TRAIN.GRAD_MAX_NORM)
    gradients, variables = zip(*grads_and_vars)
    gradients, _ = tf.clip_by_global_norm(gradients, cfg.TRAIN.GRAD_MAX_NORM)
    grads_and_vars = zip(gradients, variables)
solver_op = solver.apply_gradients(grads_and_vars)
# Save moving average of parameters
ema = tf.train.ExponentialMovingAverage(decay=cfg.TRAIN.EMA_DECAY)
ema_op = ema.apply(model.params)
with tf.control_dependencies([solver_op]):
    train_op = tf.group(ema_op)

# Save snapshot
snapshot_dir = cfg.TRAIN.SNAPSHOT_DIR % cfg.EXP_NAME
os.makedirs(snapshot_dir, exist_ok=True)
snapshot_saver = tf.train.Saver(max_to_keep=None)  # keep all snapshots
if cfg.TRAIN.START_ITER > 0:
    snapshot_file = os.path.join(snapshot_dir, f"{(cfg.TRAIN.START_ITER):08d}_{cfg.MODEL.VCR_TASK_TYPE}-{cfg.TRAIN.START_ITER}")
    print('resume training from %s' % snapshot_file)
    snapshot_saver.restore(sess, snapshot_file)
else:
    sess.run(tf.global_variables_initializer())
    if cfg.TRAIN.INIT_FROM_WEIGHTS:
        snapshot_saver.restore(sess, cfg.TRAIN.INIT_WEIGHTS_FILE)
        print('initialized from %s' % cfg.TRAIN.INIT_WEIGHTS_FILE)
# Save config
np.save(os.path.join(snapshot_dir, 'cfg.npy'), np.array(cfg))

# Write summary to TensorBoard
log_dir = cfg.TRAIN.LOG_DIR % cfg.EXP_NAME
os.makedirs(log_dir, exist_ok=True)
log_writer = tf.summary.FileWriter(log_dir, sess.graph)
loss_vqa_ph = tf.placeholder(tf.float32, [])
loss_layout_ph = tf.placeholder(tf.float32, [])
loss_total_ph = tf.placeholder(tf.float32, [])
loss_rec_ph = tf.placeholder(tf.float32, [])
accuracy_ph = tf.placeholder(tf.float32, [])
summary_trn = []
summary_trn.append(tf.summary.scalar("loss/total", loss_total_ph))
summary_trn.append(tf.summary.scalar("loss/vqa", loss_vqa_ph))
summary_trn.append(tf.summary.scalar("loss/layout", loss_layout_ph))
summary_trn.append(tf.summary.scalar("eval/vqa/accuracy", accuracy_ph))
log_step_trn = tf.summary.merge(summary_trn)

start_time = time.time()

# Initialise the profiler.
ProfileOptionBuilder = tf.profiler.ProfileOptionBuilder
profiler = tf.profiler.Profiler(sess.graph)
last_run_meta = None

sess.run(iterator.initializer)

# Run training
avg_accuracy, accuracy_decay = 0., 0.99
try:
    for n_iter in range(cfg.TRAIN.START_ITER, cfg.TRAIN.MAX_ITER):
        save_snapshot = True if ((n_iter+1) % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or (n_iter+1) == cfg.TRAIN.MAX_ITER) else False
        do_profile = True if save_snapshot and n_iter > 2498 else False

        fetches = (model.module_logits, model.vqa_scores, loss_vqa, loss_layout, loss_total, correct_label_batch, train_op)
        
        # Profile and capture metadata for this run only if we're on a specific iteration.
        run_meta = tf.compat.v1.RunMetadata() if do_profile else None
        last_run_meta = run_meta if do_profile else last_run_meta
        output = sess.run(
            fetches,
            options = tf.compat.v1.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) if do_profile else None,
            run_metadata=run_meta)

        module_logits, vqa_scores_val, loss_vqa_val, loss_layout_val, loss_total_val, vqa_q_labels, _ = output

        # compute accuracy

        if not cfg.TRAIN.SOLVER.USE_SPARSE_SOFTMAX_LABELS:
            # Reshape the expected results into a one-hot encoded vector.
            vqa_q_labels = np.reshape(vqa_q_labels, (len(vqa_q_labels) // num_answers, num_answers))

            # Get the indices of the correct answer.
            vqa_q_labels = np.where(vqa_q_labels == 1.)[1]

            # Reshape the predictions into a softmax vector.
            vqa_scores_val = np.reshape(vqa_scores_val, (len(vqa_scores_val) // num_answers, num_answers))
        
        # Convert the logits into the predicted indices of the correct answer.
        vqa_predictions = np.argmax(vqa_scores_val, axis=1)

        accuracy = np.mean(vqa_predictions == vqa_q_labels)

        avg_accuracy += (1-accuracy_decay) * (accuracy-avg_accuracy)

        # Add to TensorBoard summary
        if (n_iter+1) % cfg.TRAIN.LOG_INTERVAL == 0:
            elapsed = time.time() - start_time

            print(f"exp: {cfg.EXP_NAME}, task_type = {cfg.MODEL.VCR_TASK_TYPE}, iter = {n_iter + 1}, elapsed = {int(elapsed // 3600)}h {int(elapsed // 60) % 60}m {int(elapsed % 60)}s\n\t" +
                f"loss (vqa) = {loss_vqa_val}, loss (layout) = {loss_layout_val}, loss (total) = {loss_total_val}\n\t" +
                f"accuracy (avg) = {avg_accuracy}, accuracy (cur) = {accuracy}")
            
            summary = sess.run(log_step_trn,
                {
                    loss_total_ph: loss_total_val,
                    loss_vqa_ph: loss_vqa_val,
                    loss_layout_ph: loss_layout_val,
                    accuracy_ph: avg_accuracy
                })

            log_writer.add_summary(summary, n_iter+1)

        # Save snapshot
        if save_snapshot:
            if do_profile:
                profiler.add_step(n_iter + 1, run_meta)

                # Profile the parameters of your model.
                profiler.profile_name_scope(options=(ProfileOptionBuilder.trainable_variables_parameter()))

                # Generate an operation and memory usage timeline saved in the snapshots directory.
                profiler.profile_name_scope(options=(ProfileOptionBuilder.trainable_variables_parameter()))
                opts = (ProfileOptionBuilder(tf.profiler.ProfileOptionBuilder.time_and_memory())
                        .with_step(n_iter + 1)
                        .with_timeline_output(f'{snapshot_dir}/timeline.csv')
                        .build())

                # Generated profile can be viewed by opening a Chrome browser at https://ui.perfetto.dev/ and uploading the above timeline.csv file.
                profiler.profile_graph(options = opts)
                
                log_writer.add_run_metadata(run_metadata=run_meta, global_step=n_iter + 1, tag=f'profile_data_{n_iter + 1}')

            snapshot_file = os.path.join(snapshot_dir, f"{(n_iter + 1):08d}_{cfg.MODEL.VCR_TASK_TYPE}")
            snapshot_saver.save(sess, snapshot_file, write_meta_graph=True, global_step=n_iter+1)
            print('snapshot saved to ' + snapshot_file)
except KeyboardInterrupt:
    print('Interrupt called. Saving current checkpoint and exiting.')
    summary = sess.run(log_step_trn,
        {
            loss_vqa_ph: loss_vqa_val,
            loss_layout_ph: loss_layout_val,
            loss_total_ph: loss_total_val,
            accuracy_ph: avg_accuracy
        })

    log_writer.add_summary(summary, n_iter+1)
    snapshot_file = os.path.join(snapshot_dir, f"{(n_iter + 1):08d}_{cfg.MODEL.VCR_TASK_TYPE}")
    snapshot_saver.save(sess, snapshot_file, write_meta_graph=True, global_step=n_iter+1)
    print('Saved checkpoint. Exiting.')
    sys.exit(1)
except tf.errors.OutOfRangeError:
    print('Training iterations complete. Run profile advisor...')

# Profiler advice
ALL_ADVICE = {
    'ExpensiveOperationChecker': {},
    'AcceleratorUtilizationChecker': {},
    'JobChecker': {},  # Only available internally.
    'OperationChecker': {}
}
advice = profiler.advise(graph=sess.graph, run_meta=last_run_meta, options = ALL_ADVICE)
print('Done. Iterations complete. TF has run out of elements to train on/finished. Final results:')
print(f"exp: {cfg.EXP_NAME}, task_type = {cfg.MODEL.VCR_TASK_TYPE}, iter = {n_iter}, elapsed = {int(elapsed // 3600)}h {int(elapsed // 60) % 60}m {int(elapsed % 60)}s\n\t" +
            f"loss (vqa) = {loss_vqa_val}, loss (layout) = {loss_layout_val}, loss (total) = {loss_total_val}\n\t" +
            f"accuracy (avg) = {avg_accuracy}, accuracy (cur) = {accuracy}")