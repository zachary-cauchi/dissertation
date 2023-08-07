from tqdm import *
from urllib.parse import urlparse
import os
import h5py
import numpy as np
import copy

filesets = [
    # ('imdb_train.npy', 'bert_da_answer_train.h5', 'bert_da_rationale_train.h5'),
    # ('imdb_val.npy', 'bert_da_answer_val.h5', 'bert_da_rationale_val.h5'),
    ('imdb_test.npy', 'bert_da_answer_test.h5', 'bert_da_rationale_test.h5')
]

def get_embeddings_from_group(hgroup):
    ans = []
    ctx = []

    for subkey, dataset in hgroup.items():
        if subkey.startswith('answer_'):
            if subkey.startswith('answer_answer') or subkey.startswith('answer_rationale'):
                ans.append(np.array(dataset, np.float16))
            else:
                raise ValueError(f'Unexpected key {subkey}')
        elif subkey.startswith('ctx_'):
            ctx.append(np.array(dataset, np.float16))
        else:
            raise ValueError(f'Unexpected key {subkey}')
    return ctx, ans

for imdb_file, ans_file, rat_file in filesets:
    print(f'Loading {imdb_file}.')
    src_qars = np.load(os.path.join('imdb_r152_7x7', imdb_file), allow_pickle=True)
    print('Done, loading destination file.')
    dst_qars = np.memmap(os.path.join('imdb_bert_r152_7x7', imdb_file), dtype='object', mode='w+', shape=len(src_qars))

    is_test = 'valid_answer_index' not in src_qars[0]

    print('Processing.')
    with h5py.File(os.path.join('./bert_embeddings', ans_file), 'r') as ahf, h5py.File(os.path.join('./bert_embeddings', rat_file), 'r') as rhf:
        # Iterate over all QAR sets.
        for i, qar in enumerate(tqdm(src_qars)):
            # Look up the corresponding embeddings from the dataset.
            ans_embeddings = ahf[str(i)]
            rat_embeddings = rhf[str(i)]

            # Extract the context and answer/rationale embeddings.
            ctx_answers, answers = get_embeddings_from_group(ans_embeddings)
            ctx_rationales, rationales = get_embeddings_from_group(rat_embeddings)

            # Perform some assertions to assure ourselves the data was extracted properly.
            if is_test:
                assert len(ctx_rationales) == len(qar['all_answers']) * len(qar['all_rationales']), 'Not all combinations of answers and rationales were found.'
            else:
                assert np.shape(ctx_answers)[2] == np.shape(ctx_rationales)[2] and np.shape(ctx_answers)[0] == np.shape(ctx_rationales)[0], 'Shapes of answer and rationale contexts do not match.'

            assert np.shape(ctx_answers)[1] == len(qar['question_tokens']), 'Shapes of answer contexts do not match length of question.'

            for a, qar_a in zip(answers, qar['all_answers']):
                assert len(a) == len(qar_a), f'Answer pairing {str(a)} and {str(qar_a)} don\'t match.'

            for j, (ctx, r),  in enumerate(zip(ctx_rationales, rationales)):
                # TODO: Split the embeddings into separate question and answer contexts.
                rat_i = j % len(qar['all_rationales'])
                if is_test:
                    ans_i = int(j // len(qar['all_rationales']))
                    assert len(ctx) == len(qar['question_tokens']) + len(qar['all_answers'][ans_i]), 'Shapes of rationale contexts do not match length of question and correct answer.'
                else:
                    assert len(ctx) == len(qar['question_tokens']) + len(qar['all_answers'][qar['valid_answer_index']]), 'Shapes of rationale contexts do not match length of question and correct answer.'
                assert len(r) == len(qar['all_rationales'][rat_i]), f'Rationale pairing {str(r)} and {str(qar["all_rationales"][rat_i])} don\'t match.'

            dst_qars[i] = copy.deepcopy(qar)
            dst_qars[i]['bert_ctx_answers'] = ctx_answers
            dst_qars[i]['bert_answer_answers'] = answers
            dst_qars[i]['bert_ctx_rationales'] = ctx_rationales
            dst_qars[i]['bert_answer_rationales'] = rationales
            if i % 10000 == 0:
                dst_qars.flush()
        dst_qars.flush()
    print(f'Processing of {imdb_file} done.')
