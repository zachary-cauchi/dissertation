import argparse
import json
from collections import Counter

parser = argparse.ArgumentParser(prog='matchCorrectPredictions')
parser.add_argument('--task-type', choices=['qa', 'qar'], default='qar')
parser.add_argument('--merge-answers-file', type=argparse.FileType('r'))
parser.add_argument('--merge-direction', choices=['truth', 'predictions'])
parser.add_argument('truth_answers', type=argparse.FileType('r'))
parser.add_argument('predictions', type=argparse.FileType('r'))

args = parser.parse_args()

if args.merge_direction is not None and args.merge_answers_file is None:
    parser.print_help()
    print('merge-direction requires merge-answers-file.')
    exit(1)
elif args.merge_direction is None and args.merge_answers_file is not None:
    parser.print_help()
    print('merge-answers-file requires merge-direction.')
    exit(1)

truth_answers_path = args.truth_answers
predictions_path = args.predictions
answers_file_path = args.merge_answers_file

truth_answers = json.load(truth_answers_path)
predictions = json.load(predictions_path)

sorter_by_question = lambda x: x['question_id']
truth_answers = sorted(truth_answers, key = sorter_by_question)
predictions = sorted(predictions, key = sorter_by_question)

if answers_file_path is not None:
    answers_to_merge = json.load(answers_file_path)
    answers_to_merge = sorted(answers_to_merge, key = sorter_by_question)
    destination = truth_answers if args.merge_direction == 'truth' else predictions
    for ans, entry in zip(answers_to_merge, destination):
        entry['answer'] = ans['answer']
        entry['answer_str'] = ans['answer_str']
        if 'rationale' in ans:
            entry['rationale'] = ans['rationale']
            entry['rationale_str'] = ans['rationale_str']

    if args.merge_direction == 'truth':
        truth_answers = destination
    else:
        predictions = destination


load_rationale = args.task_type == 'qar'

correct_answers = []
correct_rationales = []
correct_predictions = []
metrics = Counter()

for i, (pred, truth) in enumerate(zip(predictions, truth_answers)):
    guessed_answer = False
    guessed_rationale = False

    if pred['answer'] == truth['answer']:
        correct_answers.append(pred)
        metrics['correct_answer'] += 1
        guessed_answer = True
    else:
        metrics['incorrect_answer'] += 1

    if load_rationale and pred['rationale'] == truth['rationale']:
        correct_rationales.append(pred)
        metrics['correct_rationale'] += 1
        guessed_rationale = True
    else:
        metrics['incorrect_rationale'] += 1

    if guessed_answer and (not load_rationale or guessed_rationale):
        correct_predictions.append(pred)
        metrics['correct_prediction'] += 1
    elif not guessed_answer and load_rationale and guessed_rationale:
        metrics['incorrect_answer_correct_rationale'] += 1
    elif guessed_answer and load_rationale and not guessed_rationale:
        metrics['correct_answer_incorrect_rationale'] += 1
    elif (not guessed_answer and not load_rationale) or (not guessed_answer and load_rationale and not guessed_rationale):
        metrics['wrong_prediction'] += 1

total_predictions = len(predictions)
total_correct_predictions = len(correct_predictions)

print(f'Evaluated {total_predictions} predictions.')
print(f'Correct answers: {metrics["correct_answer"]}, correct rationales: {metrics["correct_rationale"]}')
print(f'Correct predictions: {total_correct_predictions} in {"QA" if not load_rationale else "QAR"} mode ({(total_correct_predictions / total_predictions) * 100:.2f}%).')

if 'correct_answer_incorrect_rationale' in metrics:
    print(f'Correct answers but incorrect predictions: {metrics["correct_answer_incorrect_rationale"]} ({(metrics["correct_answer_incorrect_rationale"] / total_predictions) * 100:.2f}%)')
if 'incorrect_answer_correct_rationale' in metrics:
    print(f'Incorrect answers but correct predictions: {metrics["incorrect_answer_correct_rationale"]} ({(metrics["incorrect_answer_correct_rationale"] / total_predictions) * 100:.2f}%)')

