import time
import pprint
import tensorflow as tf
from models_vcr.config import build_cfg_from_argparse
from util.vcr_train.data_reader import DataReader

cfg = build_cfg_from_argparse()

imdb_file = cfg.IMDB_FILE % cfg.TRAIN.SPLIT_VQA
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

# Define your dataset here
dataset = data_reader.init_dataset()

# Create an iterator for the dataset
iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)

# Fetch a batch from the dataset
next_element = iterator.get_next()

max_steps = 0 # 100
with tf.compat.v1.Session() as sess:
    try:
        start_time = time.time()

        print('Sample load_one:')
        
        sample = data_reader.load_one(0)
        pprint.pprint(sample)
        
        if max_steps > 0:
            for i in range(max_steps):  # Fetch 100 batches
                curr_element = sess.run(next_element)
        else:
            while True:
                curr_element = sess.run(next_element)
    except tf.errors.OutOfRangeError:
        print("End of dataset")
    end_time = time.time()
    print("Time taken to fetch 100 batches: %s seconds" % (end_time - start_time))