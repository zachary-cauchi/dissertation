from glob import glob
from multiprocessing import pool
import sys
import os
import subprocess

# filename = 'question_answers'

if len(sys.argv) < 2:
    print(f'Usage: python {os.path.basename(__file__)} <file_prefix>')
    print(f'Example: python {os.path.basename(__file__)} question_answers')
    sys.exit(1)

filename = sys.argv[1]
input_files = glob(f'input-sentences/{filename}.*.txt')
completed_file_count = 0
total_file_count = len(input_files)

if total_file_count < 1:
    print(f'No files found matching {filename}. Exiting...')
    sys.exit(1)

def process_file(input_filename):
    global completed_file_count

    print(f'** Processing file {input_filename} **')
    basename = os.path.basename(input_filename)

    proc = subprocess.Popen(['./gen_stanford_trees.sh', input_filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Print latest progress. The parser prints the current line number in stderr so we only listen to that.
    for out in proc.stderr:
        print(f'({completed_file_count}/{total_file_count}) {basename}: {out}')

    completed_file_count += 1

    # Once completed, notify that it's completed.
    print(f'({completed_file_count}/{total_file_count}) {basename}: Processing completed')

print(f'** Processing {total_file_count} file{"s" if total_file_count > 1 else ""} **')
print(' * ' + '\n * '.join(input_files).lstrip() + '\n')

with pool.Pool(min(os.cpu_count() - 1, total_file_count)) as p:
    p.map(process_file, input_files)
    
    print('** Completed all files. Exiting **')
