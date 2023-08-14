import numpy as np
import tensorflow as tf

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_feature_list(value):
    """Returns an int64_list from a list of bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _floats_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _floats_feature_list(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature_list(value):
    """Returns a bytes_list from a list of string."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _2d_flattened_feature_list(list_2d):
    """Flatten the 2D list and get the lengths of inner lists."""
    flattened_list = [elm for sublist in list_2d for elm in sublist]
    sublist_lengths = [len(sublist) for sublist in list_2d]
    return flattened_list, sublist_lengths

def serialize_imdb_to_example(entry):
    all_answers_flattened, all_answers_length = _2d_flattened_feature_list(entry['all_answers'])
    all_answers_sequences_flattened, _ = _2d_flattened_feature_list(entry['all_answers_sequences'])
    all_rationales_flattened, all_rationales_length = _2d_flattened_feature_list(entry['all_rationales'])
    all_rationales_sequences_flattened, _ = _2d_flattened_feature_list(entry['all_rationales_sequences'])

    feature = {
        'image_name': _bytes_feature(tf.compat.as_bytes(entry['image_name'])),
        'image_path': _bytes_feature(tf.compat.as_bytes(entry['image_path'])),
        'image_id': _int64_feature(entry['image_id']),
        'feature_path': _bytes_feature(tf.compat.as_bytes(entry['feature_path'])),
        'question_id': _int64_feature(entry['question_id']),
        'question_str': _bytes_feature(tf.compat.as_bytes(entry['question_str'])),
        'question_sequence': _int64_feature_list(entry['question_sequence']),
        'question_length': _int64_feature(len(entry['question_tokens'])),
        'question_tokens': _bytes_feature_list([ tf.compat.as_bytes(token) for token in entry['question_tokens']]),
        'all_answers': _bytes_feature_list([tf.compat.as_bytes(answer) for answer in all_answers_flattened]),
        'all_answers_sequences': _int64_feature_list(all_answers_sequences_flattened),
        'all_answers_length': _int64_feature_list(all_answers_length),
        'all_rationales': _bytes_feature_list([tf.compat.as_bytes(rationale) for rationale in all_rationales_flattened]),
        'all_rationales_sequences': _int64_feature_list(all_rationales_sequences_flattened),
        'all_rationales_length': _int64_feature_list(all_rationales_length)
    }

    if 'valid_answers' in entry:
        feature['valid_answers'] = _int64_feature_list(entry['valid_answers']),
        feature['valid_answer_index'] = _int64_feature(entry['valid_answer_index']),
    if 'valid_rationales' in entry:
        feature['valid_rationales'] = _int64_feature_list(entry['valid_rationales']),
        feature['valid_rationale_index'] = _int64_feature(entry['valid_rationale_index']),

    example = tf.train.Example(features = tf.train.Features(feature=feature))
    return example.SerializeToString()

def parse_resnet_example_to_nparray(example):
    feature_description = {
        'data': tf.io.FixedLenFeature([], tf.string)
    }

    parsed_example = tf.io.parse_single_example(example, feature_description)
    return tf.io.parse_tensor(parsed_example['data'], out_type=tf.float32)

def parse_example_to_imdb_no_correct_answer(example):
    keys_to_features = {
        'image_name': tf.io.FixedLenFeature([], tf.string),
        'image_path': tf.io.FixedLenFeature([], tf.string),
        'image_id': tf.io.FixedLenFeature([], tf.int64),
        'feature_path': tf.io.FixedLenFeature([], tf.string),
        'question_id': tf.io.FixedLenFeature([], tf.int64),
        'question_str': tf.io.FixedLenFeature([], tf.string),
        'question_tokens': tf.io.VarLenFeature(tf.string),
        'question_sequence': tf.io.VarLenFeature(tf.int64),
        'question_length': tf.io.FixedLenFeature([], tf.int64),
        'all_answers': tf.io.VarLenFeature(tf.string),
        'all_answers_sequences': tf.io.VarLenFeature(tf.int64),
        'all_answers_length': tf.io.FixedLenFeature([4], tf.int64),
        'all_rationales': tf.io.VarLenFeature(tf.string),
        'all_rationales_sequences': tf.io.VarLenFeature(tf.int64),
        'all_rationales_length': tf.io.FixedLenFeature([4], tf.int64),
    }

    parsed_features = tf.io.parse_single_example(example, keys_to_features)

    image_name = tf.cast(parsed_features['image_name'], tf.string)
    image_path = tf.cast(parsed_features['image_path'], tf.string)
    feature_path = tf.cast(parsed_features['feature_path'], tf.string)
    question_str = tf.cast(parsed_features['question_str'], tf.string)
    question_sequence = tf.cast(tf.sparse.to_dense(parsed_features['question_sequence'], default_value=0), tf.int32)
    question_length = tf.cast(parsed_features['question_length'], tf.int32)
    image_id = tf.cast(parsed_features['image_id'], tf.int32)
    question_id = tf.cast(parsed_features['question_id'], tf.int32)
    question_tokens = tf.sparse.to_dense(parsed_features['question_tokens'], default_value='')
    all_answers = tf.sparse.to_dense(parsed_features['all_answers'], default_value='')
    all_answers_length = tf.cast(parsed_features['all_answers_length'], tf.int32)
    all_answers = tf.split(all_answers, all_answers_length, 0)
    all_answers_sequences = tf.cast(tf.sparse.to_dense(parsed_features['all_answers_sequences'], default_value=0), tf.int32)
    all_answers_sequences = tf.split(all_answers_sequences, all_answers_length, 0)
    all_rationales = tf.sparse.to_dense(parsed_features['all_rationales'], default_value='')
    all_rationales_length = tf.cast(parsed_features['all_rationales_length'], tf.int32)
    all_rationales = tf.split(all_rationales, all_rationales_length, 0)
    all_rationales_sequences = tf.cast(tf.sparse.to_dense(parsed_features['all_rationales_sequences'], default_value=0), tf.int32)
    all_rationales_sequences = tf.split(all_rationales_sequences, all_rationales_length, 0)

    return {
        'image_name': image_name,
        'image_path': image_path,
        'image_id': image_id,
        'feature_path': feature_path,
        'question_id': question_id,
        'question_str': question_str,
        'question_tokens': question_tokens,
        'question_sequence': question_sequence,
        'question_length': question_length,
        'all_answers': all_answers,
        'all_answers_sequences': all_answers_sequences,
        'all_answers_length': all_answers_length,
        'all_rationales': all_rationales,
        'all_rationales_sequences': all_rationales_sequences,
        'all_rationales_length': all_rationales_length
    }

def parse_example_to_imdb_with_correct_answer(example):
    keys_to_features = {
        'image_name': tf.io.FixedLenFeature([], tf.string),
        'image_path': tf.io.FixedLenFeature([], tf.string),
        'image_id': tf.io.FixedLenFeature([], tf.int64),
        'feature_path': tf.io.FixedLenFeature([], tf.string),
        'question_id': tf.io.FixedLenFeature([], tf.int64),
        'question_str': tf.io.FixedLenFeature([], tf.string),
        'question_tokens': tf.io.VarLenFeature(tf.string),
        'question_sequence': tf.io.VarLenFeature(tf.int64),
        'question_length': tf.io.FixedLenFeature([], tf.int64),
        'all_answers': tf.io.VarLenFeature(tf.string),
        'all_answers_sequences': tf.io.VarLenFeature(tf.int64),
        'all_answers_length': tf.io.FixedLenFeature([4], tf.int64),
        'all_rationales': tf.io.VarLenFeature(tf.string),
        'all_rationales_sequences': tf.io.VarLenFeature(tf.int64),
        'all_rationales_length': tf.io.FixedLenFeature([4], tf.int64),
        'valid_answers': tf.io.FixedLenFeature([4], tf.int64),
        'valid_answer_index': tf.io.FixedLenFeature([], tf.int64),
        'valid_rationales': tf.io.FixedLenFeature([4], tf.int64),
        'valid_rationale_index': tf.io.FixedLenFeature([], tf.int64),
    }

    parsed_features = tf.io.parse_single_example(example, keys_to_features)

    image_name = tf.cast(parsed_features['image_name'], tf.string)
    image_path = tf.cast(parsed_features['image_path'], tf.string)
    feature_path = tf.cast(parsed_features['feature_path'], tf.string)
    question_str = tf.cast(parsed_features['question_str'], tf.string)
    question_sequence = tf.cast(tf.sparse.to_dense(parsed_features['question_sequence'], default_value=0), tf.int32)
    question_length = tf.cast(parsed_features['question_length'], tf.int32)
    image_id = tf.cast(parsed_features['image_id'], tf.int32)
    question_id = tf.cast(parsed_features['question_id'], tf.int32)
    question_tokens = tf.sparse.to_dense(parsed_features['question_tokens'], default_value='')
    all_answers = tf.sparse.to_dense(parsed_features['all_answers'], default_value='')
    all_answers_length = tf.cast(parsed_features['all_answers_length'], tf.int32)
    all_answers = tf.split(all_answers, all_answers_length, 0)
    all_answers_sequences = tf.cast(tf.sparse.to_dense(parsed_features['all_answers_sequences'], default_value=0), tf.int32)
    all_answers_sequences = tf.split(all_answers_sequences, all_answers_length, 0)
    all_rationales = tf.sparse.to_dense(parsed_features['all_rationales'], default_value='')
    all_rationales_length = tf.cast(parsed_features['all_rationales_length'], tf.int32)
    all_rationales = tf.split(all_rationales, all_rationales_length, 0)
    all_rationales_sequences = tf.cast(tf.sparse.to_dense(parsed_features['all_rationales_sequences'], default_value=0), tf.int32)
    all_rationales_sequences = tf.split(all_rationales_sequences, all_rationales_length, 0)
    valid_answers = tf.cast(parsed_features['valid_answers'], tf.int32)
    valid_answer_index = tf.cast(parsed_features['valid_answer_index'], tf.int32)
    valid_rationales = tf.cast(parsed_features['valid_rationales'], tf.int32)
    valid_rationale_index = tf.cast(parsed_features['valid_rationale_index'], tf.int32)

    return {
        'image_name': image_name,
        'image_path': image_path,
        'image_id': image_id,
        'feature_path': feature_path,
        'question_id': question_id,
        'question_str': question_str,
        'question_tokens': question_tokens,
        'question_sequence': question_sequence,
        'question_length': question_length,
        'valid_answers': valid_answers,
        'valid_answer_index': valid_answer_index,
        'valid_rationales': valid_rationales,
        'valid_rationale_index': valid_rationale_index,
        'all_answers': all_answers,
        'all_answers_sequences': all_answers_sequences,
        'all_answers_length': all_answers_length,
        'all_rationales': all_rationales,
        'all_rationales_sequences': all_rationales_sequences,
        'all_rationales_length': all_rationales_length
    }

def serialize_bert_embeds_to_example(ctx, ans):
    ctx_flattened = [ np.reshape(c, -1) for c in ctx ]
    ans_flattened = [ np.reshape(a, -1) for a in ans ]

    ctx_features, ctx_lengths = _2d_flattened_feature_list(ctx_flattened)
    ans_features, ans_lengths = _2d_flattened_feature_list(ans_flattened)

    ctx_shapes, ctx_shapes_lengths = _2d_flattened_feature_list([ np.shape(c) for c in ctx ])
    ans_shapes, ans_shapes_lengths = _2d_flattened_feature_list([ np.shape(a) for a in ans ])

    feature = {
        'ctx': _floats_feature_list(ctx_features),
        'ans': _floats_feature_list(ans_features),
        'ctx_lengths': _int64_feature_list(ctx_lengths),
        'ans_lengths': _int64_feature_list(ans_lengths),
        'ctx_shapes': _int64_feature_list(ctx_shapes),
        'ans_shapes': _int64_feature_list(ans_shapes),
        'ctx_shapes_lengths': _int64_feature_list(ctx_shapes_lengths),
        'ans_shapes_lengths': _int64_feature_list(ans_shapes_lengths)
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

def serialize_both_bert_embeds_to_example(ctx_ans, ans, ctx_rat, rat):
    ans_example = serialize_bert_embeds_to_example(ctx_ans, ans)
    rat_example = serialize_bert_embeds_to_example(ctx_rat, rat)

    feature = {
        'ans': _bytes_feature(tf.compat.as_bytes(ans_example)),
        'rat': _bytes_feature(tf.compat.as_bytes(rat_example))
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

def parse_example_to_bert_embeds(example):
    feature_description = {
        'ctx': tf.VarLenFeature(dtype=tf.float32),
        'ans': tf.VarLenFeature(dtype=tf.float32),
        'ctx_lengths': tf.VarLenFeature(dtype=tf.int64),
        'ans_lengths': tf.VarLenFeature(dtype=tf.int64),
        'ctx_shapes': tf.VarLenFeature(dtype=tf.int64),
        'ans_shapes': tf.VarLenFeature(dtype=tf.int64),
        'ctx_shapes_lengths': tf.VarLenFeature(dtype=tf.int64),
        'ans_shapes_lengths': tf.VarLenFeature(dtype=tf.int64)
    }

    parsed_example = tf.parse_single_example(example, feature_description)

    ctx = tf.cast(tf.sparse_tensor_to_dense(parsed_example['ctx']), dtype=tf.float16)
    ans = tf.cast(tf.sparse_tensor_to_dense(parsed_example['ans']), dtype=tf.float16)
    ctx_lengths = tf.sparse_tensor_to_dense(parsed_example['ctx_lengths'])
    ans_lengths = tf.sparse_tensor_to_dense(parsed_example['ans_lengths'])
    ctx_shapes = tf.sparse_tensor_to_dense(parsed_example['ctx_shapes'])
    ans_shapes = tf.sparse_tensor_to_dense(parsed_example['ans_shapes'])

    # We have to perform some reshape trickery to get back the original structure.
    entry_count = ctx_lengths.shape[0]
    ctx_shapes = tf.reshape(ctx_shapes, [entry_count, 2])
    ans_shapes = tf.reshape(ans_shapes, [entry_count, 2])
    ctx = tf.split(ctx, ctx_lengths, 0)
    ans = tf.split(ans, ans_lengths, 0)
    for i in range(len(ans)):
        ctx[i] = tf.reshape(ctx[i], ctx_shapes[i])
        ans[i] = tf.reshape(ans[i], ans_shapes[i])

    return {
        'ctx': ctx,
        'ans': ans
    }

def parse_example_to_both_bert_embeds(example):
    feature_description = {
        'ans': tf.FixedLenFeature([], tf.string),
        'rat': tf.FixedLenFeature([], tf.string)
    }

    parsed_outer_example = tf.parse_single_example(example, feature_description)

    parsed_ans_example = parse_example_to_bert_embeds(parsed_outer_example['ans'])
    parsed_rat_example = parse_example_to_bert_embeds(parsed_outer_example['rat'])

    parsed_rat_example['rat'] = parsed_rat_example['ans']
    del parsed_rat_example['ans']

    return parsed_ans_example, parsed_rat_example
