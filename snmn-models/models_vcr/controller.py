import numpy as np
from .task_type_utils import get_name_prefix
import tensorflow as tf
from tensorflow import convert_to_tensor as to_T, newaxis as ax

from .config import cfg
from util.cnn import fc_layer as fc, fc_elu_layer as fc_elu
from util.gumbel_softmax import gumbel_softmax

import sys

class Controller:

    def __init__(self, lstm_seq, lstm_encodings, embed_seq, question_length_batch, all_answers_length_batch,
                 num_module, num_answers, scope='controller', reuse=None):
        """
        Build the controller that is used to give inputs to the neural modules.
        The controller unrolls itself for a fixed number of time steps.
        All parameters are shared across time steps.

        # The controller uses an auto-regressive structure like a GRU cell.
        # Attention is used over the input sequence.
        Here, the controller is the same as in the previous MAC paper, and
        additional module weights

        Input:
            lstm_seq: [S, N, d], tf.float32
            q_encoding: [N, d], tf.float32
            embed_seq: [S, N, e], tf.float32
            question_length_batch: [N], tf.int32
        """

        dim = cfg.MODEL.LSTM_DIM * len(lstm_encodings)
        ctrl_dim = (cfg.MODEL.EMBED_DIM if cfg.MODEL.CTRL.USE_WORD_EMBED
                    else cfg.MODEL.LSTM_DIM) * len(lstm_encodings)
        T_ctrl = cfg.MODEL.T_CTRL

        # an attention mask to normalize textual attention over the actual
        # sequence length
        S = tf.shape(lstm_seq)[0]
        N = tf.shape(lstm_seq)[1]
        # att_mask: [S, N, 1]
        # The attention mask needs to be composed of the sequence lengths of the question and each answer.

        # The total number of textual sequences to mask equal to one question + the number of answers.
        num_seqs = tf.constant(1 + num_answers, name='total_sequences')
        num_seqs_per_length = tf.divide(S, num_seqs, name='unit_seq_length')
        atts_per_seq = []

        # Create the individual attention masks for each textual sequence.
        for i in range(1 + num_answers):
            prefix = get_name_prefix(i)
            i_constant = tf.constant(i, dtype=tf.float64, name=f'{prefix}_start_index')
            i_plus_one_constant = tf.constant(i + 1, dtype=tf.float64, name=f'{prefix}_end_index')

            # Get the start and end of the textual sequence in the lstm_seq.
            # This needs to be treated as a float and then cast in the end,
            # as pre-ceiling or flooring the values will cause inaccurate lengths (too long or too short).
            S_i_start = tf.cast(tf.multiply(num_seqs_per_length, i_constant), tf.int32, name=f'{prefix}_S_i_start')
            S_i_end = tf.cast(tf.multiply(num_seqs_per_length, i_plus_one_constant), tf.int32, name=f'{prefix}_S_i_end')

            # If the current index is 0, get the question length. Otherwise, get the length of the answer at i - 1.
            seq_length = question_length_batch[:, ax] if i == 0 else all_answers_length_batch[i - 1, :, ax]

            # Build the attention mask, Setting the first n number of ints in the range equal to 1, where n = seq_length.
            att_i = tf.less(tf.range(tf.subtract(S_i_end, S_i_start, f'{prefix}_mask_length'), name=f'{prefix}_identity_attention')[:, ax, ax], seq_length, name=f'{prefix}_attention_mask')
            atts_per_seq.append(att_i)

        att_mask = tf.concat(atts_per_seq, axis=0, name='concatenated_text_attention_masks')

        # OLD IMPLEMENTATION; question only
        # att_mask = tf.less(tf.range(S)[:, ax, ax], question_length_batch[:, ax])

        att_mask = tf.cast(att_mask, tf.float32, name='final_text_attention_mask')
        with tf.variable_scope(scope, reuse=reuse):
            S = tf.shape(lstm_seq)[0]
            N = tf.shape(lstm_seq)[1]

            # manually unrolling for a number of timesteps
            c_init = tf.get_variable(
                'c_init', [1, ctrl_dim],
                initializer=tf.initializers.random_normal(
                    stddev=np.sqrt(1. / ctrl_dim)))
            c_prev = tf.tile(c_init, to_T([N, 1]))
            c_prev.set_shape([None, ctrl_dim])
            c_list = []
            cv_list = []
            module_logit_list = []
            module_prob_list = []
            for t in range(T_ctrl):

                # Prepare all question and answer encodings to combine them for the fully connected network.
                i_list = []
                for key, enc in lstm_encodings.items():
                    i_list.append(fc(f'fc_{key}_{t}', enc, output_dim=dim))  # [N, d]

                q_i = tf.concat(i_list, axis=1, name=f'fc_q_{t}')
                q_i_c = tf.concat([q_i, c_prev], axis=1)  # [N, 2d]
                cq_i = fc('fc_cq', q_i_c, output_dim=dim, reuse=(t > 0))

                # Apply a fully connected network on top of cq_i to predict the
                # module weights
                module_w_l1 = fc_elu(
                    'fc_module_w_layer1', cq_i, output_dim=dim, reuse=(t > 0))
                module_w_l2 = fc(
                    'fc_module_w_layer2', module_w_l1, output_dim=num_module,
                    reuse=(t > 0))  # [N, M]
                module_logit_list.append(module_w_l2)
                if cfg.MODEL.CTRL.USE_GUMBEL_SOFTMAX:
                    module_prob = gumbel_softmax(
                        module_w_l2, cfg.MODEL.CTRL.GUMBEL_SOFTMAX_TMP)
                else:
                    module_prob = tf.nn.softmax(module_w_l2, axis=1)
                module_prob_list.append(module_prob)

                elem_prod = tf.reshape(cq_i * lstm_seq, to_T([S*N, dim]))
                elem_prod.set_shape([None, dim])  # [S*N, d]
                raw_cv_i = tf.reshape(
                    fc('fc_cv_i', elem_prod, output_dim=1, reuse=(t > 0)),
                    to_T([S, N, 1]))
                cv_i = tf.nn.softmax(raw_cv_i, axis=0)  # [S, N, 1]
                # normalize the attention over the actual sequence length
                if cfg.MODEL.CTRL.NORMALIZE_ATT:
                    cv_i = cv_i * att_mask
                    cv_i /= tf.reduce_sum(cv_i, 0, keepdims=True)

                if cfg.MODEL.CTRL.USE_WORD_EMBED:
                    c_i = tf.reduce_sum(cv_i * embed_seq, axis=[0])  # [N, e]
                else:
                    c_i = tf.reduce_sum(cv_i * lstm_seq, axis=[0])  # [N, d]
                c_list.append(c_i)
                cv_list.append(cv_i)
                c_prev = c_i

        self.module_logits = tf.stack(module_logit_list)
        self.module_probs = tf.stack(module_prob_list)
        self.module_prob_list = module_prob_list
        self.c_list = c_list
        self.cv_list = cv_list
