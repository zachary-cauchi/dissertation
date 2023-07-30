from tqdm import *
from urllib.parse import urlparse
import os
import h5py
import numpy as np

filesets = [
    ('imdb_train.npy', 'bert_da_answer_train.h5', 'bert_da_rationale_train.h5'),
    ('imdb_val.npy', 'bert_da_answer_val.h5', 'bert_da_rationale_val.h5')
]

def parse_h5_file(filename, qar_ids, parsed_ctxs, parsed_answers):

    print(f'Parsing {h5_file}')
    with h5py.File(os.path.join('./bert_embeddings', filename), 'r') as hf:
        for key, subset in tqdm(list(hf.items())[:]):
            answers = []
            ctx = []

            qar_ids.append(int(key))
            for subkey, dataset in subset.items():
                if subkey.startswith('answer_'):
                    if subkey.startswith('answer_answer') or subkey.startswith('answer_rationale'):
                        answers.append(np.array(dataset, np.float16))
                    else:
                        raise ValueError(f'Unexpected key {subkey}')
                elif subkey.startswith('ctx_'):
                    ctx.append(np.array(dataset, np.float16))
                else:
                    raise ValueError(f'Unexpected key {subkey}')

            parsed_answers.append(answers)
            parsed_ctxs.append(ctx)

    return qar_ids, parsed_ctxs, parsed_answers

for imdb_file, ans_file, rat_file in filesets:
    imdb_qars = np.load(os.path.join('imdb_r152_7x7', imdb_file), allow_pickle=True)

    ans_qar_ids = []
    rat_qar_ids = []
    parsed_ctx_answers = []
    parsed_ctx_rationales = []
    parsed_answers = []
    parsed_rationales = []

    for ctx_dst, dst, ids, h5_file in zip([parsed_ctx_answers, parsed_ctx_rationales], [parsed_answers, parsed_rationales], [ans_qar_ids, rat_qar_ids], [ans_file, rat_file]):
        parse_h5_file(h5_file, ids, ctx_dst, dst)

    ans_sorted_qars = np.array(imdb_qars)[ans_qar_ids]
    rat_sorted_qars = np.array(imdb_qars)[rat_qar_ids]

    assert ans_qar_ids == rat_qar_ids, 'The answer and rationale embeddings are out-of-order'

    for i, (ctx_ans, ans, ctx_rat, rat, ans_sorted_qar, rat_sorted_qar) in enumerate(zip(parsed_ctx_answers, parsed_answers, parsed_ctx_rationales, parsed_rationales, ans_sorted_qars, rat_sorted_qars)):
        assert np.shape(ctx_ans)[2] == np.shape(ctx_rat)[2] and np.shape(ctx_ans)[2] == np.shape(ctx_rat)[2], 'Shapes of contexts do not match.'
        
        assert np.shape(ctx_ans)[1] == len(ans_sorted_qar['question_tokens']), 'Shapes of answer contexts do not match length of question.'

        for a, qar_a in zip(ans, ans_sorted_qar['all_answers']):
            assert len(a) == len(qar_a), f'Answer pairing {str(a)} and {str(qar_a)} don\'t match.'
        for i, (ctx, r),  in enumerate(zip(ctx_rat, rat)):
            # TODO: Split the embeddings into separate question and answer contexts.
            rat_i = i % len(rat_sorted_qar['all_rationales'])
            rat_sorted_qar['all_rationales']
            assert len(ctx) == len(rat_sorted_qar['question_tokens']) + len(rat_sorted_qar['all_answers'][rat_sorted_qar['valid_answer_index']]), 'Shapes of rationale contexts do not match length of question and correct answer.'
            assert len(r) == len(rat_sorted_qar['all_rationales'][rat_i]), f'Rationale pairing {str(r)} and {str(rat_sorted_qar["all_rationales"][rat_i])} don\'t match.'
        
        ans_sorted_qar['bert_ctx_answers'] = ctx_ans
        ans_sorted_qar['bert_answer_answers'] = ans
        rat_sorted_qar['bert_ctx_rationales'] = ctx_rat
        rat_sorted_qar['bert_answer_rationales'] = rat
    
    print('Done')