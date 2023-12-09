import argparse
import json

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

for i, (pred, truth) in enumerate(zip(predictions, truth_answers)):
    guessed_answer = False
    guessed_rationale = False

    if pred['answer'] == truth['answer']:
        correct_answers.append(pred)
        guessed_answer = True

    if load_rationale and pred['rationale'] == truth['rationale']:
        correct_rationales.append(pred)
        guessed_rationale = True

    if guessed_answer and (not load_rationale or guessed_rationale):
        correct_predictions.append(pred)

total_predictions = len(predictions)
total_correct_predictions = len(correct_predictions)

print(f'Evaluated {total_predictions} predictions.')
print(f'Correct answers: {len(correct_answers)}, correct rationales: {len(correct_rationales)}')
print(f'Correct predictions: {total_correct_predictions} in {"QA" if not load_rationale else "QAR"} mode ({(total_correct_predictions / total_predictions) * 100:.2f}%).')
if load_rationale:
    print(f'Correct answers but incorrect predictions')