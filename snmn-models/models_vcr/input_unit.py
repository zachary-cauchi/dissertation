import numpy as np
import tensorflow as tf
from tensorflow import convert_to_tensor as to_T

from .config import cfg
from util.cnn import conv_elu_layer as conv_elu, conv_layer as conv

# TODO: Fix method comment block.
def build_input_unit(input_seq_batch, all_answers_seq_batch, seq_length_batch, all_answers_seq_length_batch, num_vocab, num_answers,
                     scope='input_unit', reuse=None):
    """
    Preprocess the input sequence with a (single-layer) bidirectional LSTM.

    Input:
        input_seq_batch: [S, N], tf.int32
        seq_length_batch: [N], tf.int32
    Return:
        lstm_seq: [S, N, d], tf.float32
        q_encoding: [N, d], tf.float32
        embed_seq: [S, N, e], tf.float32
    """

    with tf.variable_scope(scope, reuse=reuse):
        # word embedding
        embed_dim = cfg.MODEL.EMBED_DIM
        if cfg.USE_FIXED_WORD_EMBED:
            embed_mat = to_T(np.load(cfg.FIXED_WORD_EMBED_FILE), name='word_embeddings_tensor')
        else:
            embed_mat = tf.get_variable(
                'embed_mat', [num_vocab, embed_dim],
                initializer=tf.initializers.random_normal(
                    stddev=np.sqrt(1. / embed_dim)))

        # bidirectional LSTM
        lstm_dim = cfg.MODEL.LSTM_DIM
        assert lstm_dim % 2 == 0, \
            'lstm_dim is the dimension of [fw, bw] and must be a multiply of 2'

        lstm_outs = ()
        lstm_encs = {}

        # For the question, and each answer, generate lstms.
        for i in range(1 + num_answers):
            prefix = 'question' if i == 0 else f'answer{i}'

            if i == 0:
                embed_seq = tf.nn.embedding_lookup(embed_mat, input_seq_batch, prefix + '_word_embeddings_lookup')
            else:
                # Load the i-1'th answer from the input and generate it's embedding.
                answer_seq_batch = tf.gather_nd(indices=[i-1], params=all_answers_seq_batch, name='get_' + prefix)

                embed_seq = tf.nn.embedding_lookup(embed_mat, answer_seq_batch, prefix + '_word_embeddings_lookup')

            # Casting required when using generated weights.
            embed_seq = tf.cast(embed_seq, tf.float32, name=prefix + '_cast_embeds_to_32')

            cell_fw = tf.nn.rnn_cell.LSTMCell(lstm_dim//2, name=prefix + '_fw_lstm_cell')
            cell_bw = tf.nn.rnn_cell.LSTMCell(lstm_dim//2, name=prefix + '_bw_lstm_cell')
            
            # Create the lstm, getting the output and their states.
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, inputs=embed_seq, dtype=tf.float32,
                sequence_length=seq_length_batch if i == 0 else all_answers_seq_length_batch[i - 1],
                time_major=True)

            # concatenate the final hidden state of the forward and backward LSTM
            # for question (or answer) representation
            seq_encoding = tf.concat([states[0].h, states[1].h], axis=1, name=prefix + '_create_encoded_representation')

            lstm_outs = lstm_outs + outputs
            lstm_encs[prefix] = seq_encoding

        # concatenate the hidden state from forward and backward LSTM
        lstm_seq = tf.concat(lstm_outs, axis=2, name=f'hidden_lstm_seq')

    return lstm_seq, lstm_encs, embed_seq


def get_positional_encoding(H, W):
    pe_dim = cfg.MODEL.PE_DIM
    assert pe_dim % 4 == 0, 'pe_dim must be a multiply of 4 (h/w x sin/cos)'
    c_period = 10000. ** np.linspace(0., 1., pe_dim // 4)
    h_vec = np.tile(np.arange(0, H).reshape((H, 1, 1)), (1, W, 1)) / c_period
    w_vec = np.tile(np.arange(0, W).reshape((1, W, 1)), (H, 1, 1)) / c_period
    position_encoding = np.concatenate(
        (np.sin(h_vec), np.cos(h_vec), np.sin(w_vec), np.cos(w_vec)), axis=-1)
    position_encoding = position_encoding.reshape((1, H, W, pe_dim))
    return position_encoding


def build_kb_batch(image_feat_batch, scope='kb_batch', reuse=None):
    """
    Concatenation image batch and position encoding batch, and apply a 2-layer
    CNN on top of it.

    Input:
        image_feat_batch: [N, H, W, C], tf.float32
    Return:
        kb_batch: [N, H, W, d], tf.float32
    """

    kb_dim = cfg.MODEL.KB_DIM
    with tf.variable_scope(scope, reuse=reuse):
        if cfg.MODEL.INPUT.USE_L2_NORMALIZATION:
            norm_type = cfg.MODEL.INPUT.L2_NORMALIZATION_TYPE
            if norm_type == 'global':
                # Normalize along H, W, C
                image_feat_batch = tf.nn.l2_normalize(
                    image_feat_batch, axis=[1, 2, 3])
            elif norm_type == 'local':
                # Normalize along C
                image_feat_batch = tf.nn.l2_normalize(
                    image_feat_batch, axis=-1)
            else:
                raise ValueError('Invalid l2 normalization type: ' + norm_type)

        if cfg.MODEL.INPUT.USE_POSITION_ENCODING:
            # get positional encoding
            N = tf.shape(image_feat_batch)[0]

            _, H, W, _ = image_feat_batch.get_shape().as_list()
            position_encoding = to_T(
                get_positional_encoding(H, W), dtype=tf.float32)
            position_batch = tf.tile(position_encoding, to_T([N, 1, 1, 1]))

            # apply a two layer convnet with ELU activation
            conv1 = conv_elu(
                'conv1', tf.concat([image_feat_batch, position_batch], axis=3),
                kernel_size=1, stride=1, output_dim=kb_dim)
            conv2 = conv(
                'conv2', conv1, kernel_size=1, stride=1, output_dim=kb_dim)

            kb_batch = conv2
        else:
            kb_batch = conv('conv_no_pe', image_feat_batch, kernel_size=1,
                            stride=1, output_dim=kb_dim)
    return kb_batch
