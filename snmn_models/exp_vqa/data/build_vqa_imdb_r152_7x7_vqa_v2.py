import numpy as np
import json
import os

import sys; sys.path.append('../../')  # NOQA
from util import text_processing
from collections import Counter

vocab_answer_file = './answers_vqa.txt'
annotation_file = '../vqa_dataset/Annotations/v2_mscoco_%s_annotations.json'
question_file = '../vqa_dataset/Questions/v2_OpenEnded_mscoco_%s_questions.json'
gt_layout_file = './v2_gt_layout_%s_new_parse.npy'

image_dir = '../coco_dataset/images/%s/'
feature_dir = './resnet152_c5_7x7/%s/'

answer_dict = text_processing.VocabDict(vocab_answer_file)
valid_answer_set = set(answer_dict.word_list)


def extract_answers(q_answers):
    all_answers = [answer["answer"] for answer in q_answers]
    valid_answers = [a for a in all_answers if a in valid_answer_set]
    # build soft scores
    soft_score_inds = []
    soft_score_target = []
    valid_answer_counter = Counter(valid_answers)
    for k, v in valid_answer_counter.items():
        soft_score_inds.append(answer_dict.word2idx(k))
        soft_score_target.append(min(1., v / 3.))
    return all_answers, valid_answers, soft_score_inds, soft_score_target


def build_imdb(image_set):
    print('building imdb %s' % image_set)
    if image_set in ['train2014', 'val2014']:
        load_answer = True
        load_gt_layout = True
        with open(annotation_file % image_set) as f:
            annotations = json.load(f)["annotations"]
            qid2ann_dict = {ann['question_id']: ann for ann in annotations}
        qid2layout_dict = np.load(gt_layout_file % image_set, allow_pickle=True)[()]
    else:
        load_answer = False
        load_gt_layout = False
    with open(question_file % image_set) as f:
        questions = json.load(f)['questions']
    coco_set_name = image_set.replace('-dev', '')
    abs_image_dir = os.path.abspath(image_dir % coco_set_name)
    abs_feature_dir = os.path.abspath(feature_dir % coco_set_name)
    image_name_template = 'COCO_' + coco_set_name + '_%012d'
    imdb = [None]*len(questions)

    unk_ans_count = 0
    for n_q, q in enumerate(questions):
        if (n_q+1) % 10000 == 0:
            print('processing %d / %d' % (n_q+1, len(questions)))
        image_id = q['image_id']
        question_id = q['question_id']
        image_name = image_name_template % image_id
        image_path = os.path.join(abs_image_dir, image_name + '.jpg')
        feature_path = os.path.join(abs_feature_dir, image_name + '.npy')
        question_str = q['question']
        question_tokens = text_processing.tokenize(question_str)

        iminfo = dict(image_name=image_name,
                      image_path=image_path,
                      image_id=image_id,
                      question_id=question_id,
                      feature_path=feature_path,
                      question_str=question_str,
                      question_tokens=question_tokens)

        # load answers
        if load_answer:
            ann = qid2ann_dict[question_id]
            all_answers, valid_answers, soft_score_inds, soft_score_target = \
                extract_answers(ann['answers'])
            if len(valid_answers) == 0:
                valid_answers = ['<unk>']
                unk_ans_count += 1
            iminfo['all_answers'] = all_answers
            iminfo['valid_answers'] = valid_answers
            iminfo['soft_score_inds'] = soft_score_inds
            iminfo['soft_score_target'] = soft_score_target

        if load_gt_layout:
            gt_layout_tokens = qid2layout_dict[question_id]
            iminfo['gt_layout_tokens'] = gt_layout_tokens

        imdb[n_q] = iminfo
    print('%d out of %d answers are <unk>' % (unk_ans_count, len(questions)))
    return imdb


imdb_train2014 = build_imdb('train2014')
imdb_val2014 = build_imdb('val2014')
imdb_test2015 = build_imdb('test2015')
imdb_test_dev2015 = build_imdb('test-dev2015')

os.makedirs('./imdb_r152_7x7_vqa_v2', exist_ok=True)
np.save('./imdb_r152_7x7_vqa_v2/imdb_v2_train2014.npy', np.array(imdb_train2014))
np.save('./imdb_r152_7x7_vqa_v2/imdb_v2_val2014.npy', np.array(imdb_val2014))
np.save('./imdb_r152_7x7_vqa_v2/imdb_v2_trainval2014.npy', np.array(imdb_train2014+imdb_val2014))
np.save('./imdb_r152_7x7_vqa_v2/imdb_v2_test2015.npy', np.array(imdb_test2015))
np.save('./imdb_r152_7x7_vqa_v2/imdb_v2_test-dev2015.npy', np.array(imdb_test_dev2015))
