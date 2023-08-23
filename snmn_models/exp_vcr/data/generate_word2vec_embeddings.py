import argparse
import numpy as np
from gensim.models import word2vec

parser = argparse.ArgumentParser(prog='generate_word2vec_embeddings.py', description='Generate Word2Vec embeddings for the VCR corpus.')
parser.add_argument('--file_path', type=str, default='./corpus_vcr.txt')
parser.add_argument('--model_type', type=str, choices=['cbow', 'sg'], default='cbow')
parser.add_argument('--vocab_file', type=str, default='./vocabulary_vcr.txt')
parser.add_argument('--output_file', type=str, default='./vocabulary_vcr_word2vec.txt')
parser.add_argument('--output_npy_file', type=str, default='./vocabulary_vcr_word2vec.npy')

args = parser.parse_args()


model = word2vec.Word2Vec(corpus_file=args.file_path, min_count=1, vector_size=300, sg = 1 if args.model_type == 'sg' else 0)

with open(args.vocab_file, 'r') as vocab_file:
    vocab_words = [ line.strip().split(' ')[0] for line in vocab_file ]

assert len(vocab_words) == len(model.wv), f'The vocabulary and model vocabulary do not match. The vocabulary file is {len(vocab_words)} tokens long but the models vocabulary is {len(model.vw)} long.'

vectors_array = np.array([ model.wv[word] for word in vocab_words ])

assert np.array_equal(vectors_array[5], model.wv[vocab_words[5]]), 'numpy array and model array do not share the same order.'

np.save(args.output_npy_file, vectors_array, allow_pickle=True)

with open(args.output_file, 'w') as out_txt:
    for vector in vectors_array:
        vector_str = ' '.join(map(str, vector))
        out_txt.write(vector_str + '\n')