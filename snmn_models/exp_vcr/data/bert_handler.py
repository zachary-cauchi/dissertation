import numpy as np
import h5py

class BertHandler:
    def __init__(self, bert_answer_embeddings_path: str, bert_rationale_embeddings_path: str, load_correct_answer: bool = True):
        self.load_correct_answer = load_correct_answer
        self.ans_hf = h5py.File(bert_answer_embeddings_path, mode='r')
        self.rat_hf = h5py.File(bert_rationale_embeddings_path, mode='r')
        self.bert_dim = len(self.ans_hf['0']['answer_answer0'][0])

    def __del__(self):
        if hasattr(self, 'ans_hf'):
            self.ans_hf.close()
        if hasattr(self, 'rat_hf'):
            self.rat_hf.close()

    def get_embeddings_by_id(self, id):
        return  {
            'ans': self.get_embeddings_from_group(self.ans_hf[str(id)]),
            'rat': self.get_embeddings_from_group(self.rat_hf[str(id)])
        }

    def get_embeddings_from_group(self, hgroup):
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

    def validate_embeddings(self, ctx_answers, ctx_rationales, answers, rationales, qar):
        if not self.load_correct_answer:
            assert len(ctx_rationales) == len(qar['all_answers']) * len(qar['all_rationales']), 'Not all combinations of answers and rationales were found.'
        else:
            assert np.shape(ctx_answers)[2] == np.shape(ctx_rationales)[2] and np.shape(ctx_answers)[0] == np.shape(ctx_rationales)[0], 'Shapes of answer and rationale contexts do not match.'

        assert np.shape(ctx_answers)[1] == len(qar['question_tokens']), 'Shapes of answer contexts do not match length of question.'

        for a, qar_a in zip(answers, qar['all_answers']):
            assert len(a) == len(qar_a), f'Answer pairing {str(a)} and {str(qar_a)} don\'t match.'

        for j, (ctx, r),  in enumerate(zip(ctx_rationales, rationales)):
            rat_i = j % len(qar['all_rationales'])
            if not self.load_correct_answer:
                ans_i = int(j // len(qar['all_rationales']))
                assert len(ctx) == len(qar['question_tokens']) + len(qar['all_answers'][ans_i]), 'Shapes of rationale contexts do not match length of question and correct answer.'
            else:
                assert len(ctx) == len(qar['question_tokens']) + len(qar['all_answers'][qar['valid_answer_index']]), 'Shapes of rationale contexts do not match length of question and correct answer.'
            assert len(r) == len(qar['all_rationales'][rat_i]), f'Rationale pairing {str(r)} and {str(qar["all_rationales"][rat_i])} don\'t match.'