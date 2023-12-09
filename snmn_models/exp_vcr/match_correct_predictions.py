import argparse
import json
from collections import Counter

parser = argparse.ArgumentParser(prog='matchCorrectPredictions')
parser.add_argument('--task-type', choices=['qa', 'qar'], default='qar')
parser.add_argument('truth_answers', type=argparse.FileType('r'))
parser.add_argument('predictions', type=argparse.FileType('r'))

args = parser.parse_args()

truth_answers_path = args.truth_answers
predictions_path = args.predictions

truth_answers = json.load(truth_answers_path)
predictions = json.load(predictions_path)

truth_answers = sorted(truth_answers, key = lambda x: x['question_id'])
predictions = sorted(predictions, key = lambda x: x['question_id'])

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

    if load_rationale and pred['rationale'] == truth['rationale']:
        correct_rationales.append(pred)
        metrics['correct_rationale'] += 1
        guessed_rationale = True

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

