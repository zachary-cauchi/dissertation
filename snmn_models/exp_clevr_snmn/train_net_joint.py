import os
import numpy as np
import tensorflow as tf

from models_clevr_snmn.model import Model
from models_clevr_snmn.config import build_cfg_from_argparse
from util.clevr_train.data_reader import DataReader
from util import boxes
from util.losses import SharpenLossScaler

# Load config
cfg = build_cfg_from_argparse()

# Start session
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPU_ID)
sess = tf.Session(config=tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=cfg.GPU_MEM_GROWTH)))

# Data files
imdb_file_vqa = cfg.IMDB_FILE % cfg.TRAIN.SPLIT_VQA
imdb_file_loc = cfg.IMDB_FILE % cfg.TRAIN.SPLIT_LOC
data_reader_vqa = DataReader(
    imdb_file_vqa, shuffle=True, one_pass=False,
    batch_size=cfg.TRAIN.BATCH_SIZE,
    vocab_question_file=cfg.VOCAB_QUESTION_FILE, T_encoder=cfg.MODEL.T_ENCODER,
    vocab_answer_file=cfg.VOCAB_ANSWER_FILE, load_gt_layout=True,
    vocab_layout_file=cfg.VOCAB_LAYOUT_FILE, T_decoder=cfg.MODEL.T_CTRL)
data_reader_loc = DataReader(
    imdb_file_loc, shuffle=True, one_pass=False,
    batch_size=cfg.TRAIN.BATCH_SIZE,
    vocab_question_file=cfg.VOCAB_QUESTION_FILE, T_encoder=cfg.MODEL.T_ENCODER,
    vocab_answer_file=cfg.VOCAB_ANSWER_FILE, load_gt_layout=True,
    vocab_layout_file=cfg.VOCAB_LAYOUT_FILE, T_decoder=cfg.MODEL.T_CTRL,
    img_H=cfg.MODEL.H_IMG, img_W=cfg.MODEL.W_IMG)
num_vocab = data_reader_vqa.batch_loader.vocab_dict.num_vocab
num_choices = data_reader_vqa.batch_loader.answer_dict.num_vocab
module_names = data_reader_vqa.batch_loader.layout_dict.word_list

# Inputs and model
input_seq_batch = tf.placeholder(tf.int32, [None, None])
seq_length_batch = tf.placeholder(tf.int32, [None])
image_feat_batch = tf.placeholder(
    tf.float32, [None, cfg.MODEL.H_FEAT, cfg.MODEL.W_FEAT, cfg.MODEL.FEAT_DIM])
model = Model(
    input_seq_batch, seq_length_batch, image_feat_batch, num_vocab=num_vocab,
    num_choices=num_choices, module_names=module_names, is_training=True)

# Loss function
is_vqa_batch = tf.placeholder(tf.bool, [])
answer_label_batch = tf.placeholder(tf.int32, [None])
loss_vqa = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=model.vqa_scores, labels=answer_label_batch))
loss_vqa = tf.where(is_vqa_batch, loss_vqa, 0)
bbox_ind_batch = tf.placeholder(tf.int32, [None])
bbox_offset_batch = tf.placeholder(tf.float32, [None, 4])
loss_bbox_ind = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=model.loc_scores, labels=bbox_ind_batch))
loss_bbox_offset = model.bbox_offset_loss(bbox_ind_batch, bbox_offset_batch)
loss_bbox_ind = tf.where(tf.logical_not(is_vqa_batch), loss_bbox_ind, 0)
loss_bbox_offset = tf.where(tf.logical_not(is_vqa_batch), loss_bbox_offset, 0)
if cfg.TRAIN.USE_GT_LAYOUT:
    gt_layout_batch = tf.placeholder(tf.int32, [None, None])
    loss_layout = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=model.module_logits, labels=gt_layout_batch))
else:
    loss_layout = tf.convert_to_tensor(0.)
loss_rec = model.rec_loss
sharpen_scale_ph = tf.placeholder(tf.float32, [])
if cfg.TRAIN.USE_SHARPEN_LOSS:
    loss_sharpen = model.sharpen_loss()
else:
    loss_sharpen = tf.convert_to_tensor(0.)
loss_train = (loss_vqa * cfg.TRAIN.VQA_LOSS_WEIGHT +
              loss_bbox_ind * cfg.TRAIN.BBOX_IND_LOSS_WEIGHT +
              loss_bbox_offset * cfg.TRAIN.BBOX_OFFSET_LOSS_WEIGHT +
              loss_layout * cfg.TRAIN.LAYOUT_LOSS_WEIGHT +
              loss_rec * cfg.TRAIN.REC_LOSS_WEIGHT +
              loss_sharpen * cfg.TRAIN.SHARPEN_LOSS_WEIGHT * sharpen_scale_ph)
loss_total = loss_train + cfg.TRAIN.WEIGHT_DECAY * model.l2_reg

# Train with Adam
solver = tf.train.AdamOptimizer(learning_rate=cfg.TRAIN.SOLVER.LR)
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
    snapshot_file = os.path.join(snapshot_dir, "%08d" % cfg.TRAIN.START_ITER)
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
log_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
loss_vqa_ph = tf.placeholder(tf.float32, [])
loss_bbox_ind_ph = tf.placeholder(tf.float32, [])
loss_bbox_offset_ph = tf.placeholder(tf.float32, [])
loss_layout_ph = tf.placeholder(tf.float32, [])
loss_rec_ph = tf.placeholder(tf.float32, [])
loss_sharpen_ph = tf.placeholder(tf.float32, [])
vqa_accuracy_ph = tf.placeholder(tf.float32, [])
loc_accuracy_ph = tf.placeholder(tf.float32, [])
summary_trn = []
summary_trn.append(tf.summary.scalar("loss/vqa", loss_vqa_ph))
summary_trn.append(tf.summary.scalar("loss/bbox_ind", loss_bbox_ind_ph))
summary_trn.append(tf.summary.scalar("loss/bbox_offset", loss_bbox_offset_ph))
summary_trn.append(tf.summary.scalar("loss/layout", loss_layout_ph))
summary_trn.append(tf.summary.scalar("loss/rec", loss_rec_ph))
summary_trn.append(tf.summary.scalar("loss/sharpen", loss_sharpen_ph))
summary_trn.append(tf.summary.scalar("loss/sharpen_scale", sharpen_scale_ph))
summary_trn.append(tf.summary.scalar("eval/vqa/accuracy", vqa_accuracy_ph))
summary_trn.append(tf.summary.scalar("eval/loc/P1", loc_accuracy_ph))
log_step_trn = tf.summary.merge(summary_trn)

# Run training
vqa_avg_accuracy, loc_avg_accuracy, accuracy_decay = 0., 0., 0.99
iou_th = cfg.TRAIN.BBOX_IOU_THRESH
sharpen_loss_scaler = SharpenLossScaler(cfg)
for n_batch, (batch_vqa, batch_loc) in enumerate(
        zip(data_reader_vqa.batches(), data_reader_loc.batches())):
    n_iter = n_batch + cfg.TRAIN.START_ITER
    if n_iter >= cfg.TRAIN.MAX_ITER:
        break

    sharpen_scale = sharpen_loss_scaler(n_iter)
    feed_dict = {input_seq_batch: batch_vqa['input_seq_batch'],
                 seq_length_batch: batch_vqa['seq_length_batch'],
                 image_feat_batch: batch_vqa['image_feat_batch'],
                 answer_label_batch: batch_vqa['answer_label_batch'],
                 bbox_ind_batch: np.zeros(
                    len(batch_vqa['image_feat_batch']), np.int32),
                 bbox_offset_batch: np.zeros(
                    (len(batch_vqa['image_feat_batch']), 4), np.float32),
                 is_vqa_batch: True,
                 sharpen_scale_ph: sharpen_scale}
    if cfg.TRAIN.USE_GT_LAYOUT:
        feed_dict[gt_layout_batch] = batch_vqa['gt_layout_batch']
    vqa_scores_val, loss_vqa_val, loss_layout_val1, _ = sess.run(
        (model.vqa_scores, loss_vqa, loss_layout, train_op), feed_dict)

    feed_dict = {input_seq_batch: batch_loc['input_seq_batch'],
                 seq_length_batch: batch_loc['seq_length_batch'],
                 image_feat_batch: batch_loc['image_feat_batch'],
                 answer_label_batch: np.zeros(
                    len(batch_loc['image_feat_batch']), np.int32),
                 bbox_ind_batch: batch_loc['bbox_ind_batch'],
                 bbox_offset_batch: batch_loc['bbox_offset_batch'],
                 is_vqa_batch: False,
                 sharpen_scale_ph: sharpen_scale}
    if cfg.TRAIN.USE_GT_LAYOUT:
        feed_dict[gt_layout_batch] = batch_loc['gt_layout_batch']
    loc_scores_val, bbox_offset_val, loss_bbox_ind_val, loss_bbox_offset_val, \
        loss_layout_val2, loss_rec_val, loss_sharpen_val, _ = sess.run(
            (model.loc_scores, model.bbox_offset, loss_bbox_ind,
             loss_bbox_offset, loss_layout, loss_rec, loss_sharpen, train_op),
            feed_dict)

    # compute accuracy
    vqa_labels = batch_vqa['answer_label_batch']
    vqa_predictions = np.argmax(vqa_scores_val, axis=1)
    vqa_accuracy = np.mean(vqa_predictions == vqa_labels)
    vqa_avg_accuracy += (1-accuracy_decay) * (vqa_accuracy-vqa_avg_accuracy)

    bbox_pred = boxes.batch_feat_grid2bbox(
        np.argmax(loc_scores_val, axis=1), bbox_offset_val,
        data_reader_loc.batch_loader.stride_H,
        data_reader_loc.batch_loader.stride_W,
        data_reader_loc.batch_loader.feat_H,
        data_reader_loc.batch_loader.feat_W)
    bbox_gt = batch_loc['bbox_batch']
    loc_accuracy = np.mean(boxes.batch_bbox_iou(bbox_pred, bbox_gt) >= iou_th)
    loc_avg_accuracy += (1-accuracy_decay) * (loc_accuracy-loc_avg_accuracy)

    # Add to TensorBoard summary
    if (n_iter+1) % cfg.TRAIN.LOG_INTERVAL == 0:
        loss_layout_val = (loss_layout_val1 + loss_layout_val2) / 2.
        print("exp: %s, iter = %d\n\t" % (cfg.EXP_NAME, n_iter+1) +
              "loss (vqa) = %f, loss (bbox_ind) = %f, "
              "loss (bbox_offset) = %f, loss (layout) = %f, loss (rec) = %f, "
              "loss (sharpen) = %f, sharpen_scale = %f\n\t" % (
                loss_vqa_val, loss_bbox_ind_val, loss_bbox_offset_val,
                loss_layout_val, loss_rec_val, loss_sharpen_val,
                sharpen_scale) +
              "accuracy (vqa, cur) = %f, accuracy (vqa, avg) = %f\n\t" % (
               vqa_accuracy, vqa_avg_accuracy) +
              "P1@%.2f (loc, cur) = %f, P1@%.2f (loc, avg) = %f" % (
               iou_th, loc_accuracy, iou_th, loc_avg_accuracy))
        summary = sess.run(log_step_trn, {
            loss_vqa_ph: loss_vqa_val,
            loss_bbox_ind_ph: loss_bbox_ind_val,
            loss_bbox_offset_ph: loss_bbox_offset_val,
            loss_layout_ph: loss_layout_val,
            loss_rec_ph: loss_rec_val,
            loss_sharpen_ph: loss_sharpen_val, sharpen_scale_ph: sharpen_scale,
            vqa_accuracy_ph: vqa_avg_accuracy,
            loc_accuracy_ph: loc_avg_accuracy})
        log_writer.add_summary(summary, n_iter+1)

    # Save snapshot
    if ((n_iter+1) % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or
            (n_iter+1) == cfg.TRAIN.MAX_ITER):
        snapshot_file = os.path.join(snapshot_dir, "%08d" % (n_iter+1))
        snapshot_saver.save(sess, snapshot_file, write_meta_graph=False)
        print('snapshot saved to ' + snapshot_file)
