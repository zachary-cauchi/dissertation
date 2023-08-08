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

def parse_example_to_imdb(example):
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
    question_sequence = tf.sparse.to_dense(parsed_features['question_sequence'], default_value=0)
    question_length = tf.cast(parsed_features['question_length'], tf.int32)
    image_id = tf.cast(parsed_features['image_id'], tf.int32)
    question_id = tf.cast(parsed_features['question_id'], tf.int32)
    question_tokens = tf.sparse.to_dense(parsed_features['question_tokens'], default_value='')
    valid_answers = tf.cast(parsed_features['valid_answers'], tf.int32)
    valid_answer_index = tf.cast(parsed_features['valid_answer_index'], tf.int32)
    valid_rationales = tf.cast(parsed_features['valid_rationales'], tf.int32)
    valid_rationale_index = tf.cast(parsed_features['valid_rationale_index'], tf.int32)
    all_answers = tf.sparse.to_dense(parsed_features['all_answers'], default_value='')
    all_answers_length = tf.cast(parsed_features['all_answers_length'], tf.int32)
    all_answers = tf.split(all_answers, all_answers_length, 0)
    all_answers_sequences = tf.sparse.to_dense(parsed_features['all_answers_sequences'], default_value=0)
    all_answers_sequences = tf.split(all_answers_sequences, all_answers_length, 0)
    all_rationales = tf.sparse.to_dense(parsed_features['all_rationales'], default_value='')
    all_rationales_length = tf.cast(parsed_features['all_rationales_length'], tf.int32)
    all_rationales = tf.split(all_rationales, all_rationales_length, 0)
    all_rationales_sequences = tf.sparse.to_dense(parsed_features['all_rationales_sequences'], default_value=0)
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