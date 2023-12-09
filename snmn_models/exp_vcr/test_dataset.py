import time
import os
import json
import tensorflow as tf
from models_vcr.config import build_cfg_from_argparse
from util.vcr_train.data_reader import DataReader

cfg = build_cfg_from_argparse()

split_vqa = cfg.TEST.SPLIT_VQA
imdb_file = cfg.IMDB_FILE % split_vqa
data_reader = DataReader(
    imdb_file, shuffle=True, one_pass=False, batch_size=cfg.TRAIN.BATCH_SIZE,
    vocab_question_file=cfg.VOCAB_QUESTION_FILE,
    T_q_encoder=cfg.MODEL.T_Q_ENCODER,
    T_a_encoder=cfg.MODEL.T_A_ENCODER,
    T_r_encoder=cfg.MODEL.T_R_ENCODER,
    vocab_answer_file=cfg.VOCAB_ANSWER_FILE % split_vqa,
    load_gt_layout=cfg.TRAIN.USE_GT_LAYOUT,
    vocab_layout_file=cfg.VOCAB_LAYOUT_FILE, T_decoder=cfg.MODEL.T_CTRL,
    load_soft_score=cfg.TRAIN.VQA_USE_SOFT_SCORE,
    feed_answers_with_input=cfg.MODEL.INPUT.USE_ANSWERS,
    vcr_task_type=cfg.MODEL.VCR_TASK_TYPE,
    use_sparse_softmax_labels=cfg.TRAIN.SOLVER.USE_SPARSE_SOFTMAX_LABELS,
    load_bert_embeddings = cfg.USE_BERT_SENTENCE_EMBED,
    bert_embeddings_path = os.path.join(cfg.BERT_EMBED_DIR, split_vqa),
    external_true_answers_file = cfg.TRAIN.EXTERNAL_TRUE_ANSWERS_FILE if cfg.TRAIN.USE_EXTERNAL_TRUE_ANSWERS_FILE else ''
)

# Define your dataset here
dataset = data_reader.init_dataset()

max_epochs = 1

max_steps = 0 # 100
with tf.compat.v1.Session() as sess, open(split_vqa + '_true_answers.json', 'w') as f:
    for epoch in range(max_epochs):
        print(f'Measuring epoch {epoch}')
        # Create an iterator for the dataset
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        qars = []

        # Fetch a batch from the dataset
        next_element = iterator.get_next()
        try:
            start_time = time.time()

            if max_steps > 0:
                for j in range(max_steps):  # Fetch 100 batches
                    curr_element = sess.run(next_element)
            else:
                while True:
                    curr_element = sess.run(next_element)

                    qar_record = curr_element[0]
                    qar_record['all_answers'] = qar_record['all_answers'].tolist()
                    qar_record['all_rationales'] = qar_record['all_rationales'].tolist()
                    qar_record['valid_answer_index'] = qar_record['valid_answer_index'].tolist()
                    qar_record['valid_rationale_index'] = qar_record['valid_rationale_index'].tolist()
                    qar_sentences = []

                    # Only use every nth record because the others will repeat the same thing.
                    for i in range(0, len(qar_record['question_str']), data_reader.num_combinations):
                        qar = {
                            'image_id': int(qar_record['image_id'][i]),
                            'question_id': int(qar_record['question_id'][i]),
                            'question': qar_record['question_str'][i].decode(),
                        }
                        qar['answer'] = qar_record['valid_answer_index'][i]
                        qar['answer_str'] = ' '.join([ token.decode() for token in qar_record['all_answers'][i] if len(token) > 0 ])
                        ri = i + qar_record['valid_rationale_index'][i]
                        qar['rationale'] = qar_record['valid_rationale_index'][i]
                        qar['rationale_str'] = ' '.join([ token.decode() for token in qar_record['all_rationales'][ri] if len(token) > 0 ])
                        qar_sentences.append(qar)

                    qars.extend(qar_sentences)
        except tf.errors.OutOfRangeError:
            print(f'Epoch {epoch} finished: End of dataset')
            qars = sorted(qars, key = lambda x: x['question_id'])
            json.dump(qars, f, indent = 2)
            f.flush()
        end_time = time.time()

    print(f'Last fetched sample: {curr_element}')

    print(f'Time taken to fetch {"all" if max_steps == 0 else max_steps} batch{"es" if max_steps != 1 else ""} for epoch {i}: {end_time - start_time} seconds')
