#!/usr/bin/env bash
#
# Runs the run_parser.sh file on all questions, answers, and rationales.
# A file with the name of the current batch being processed will be present until that batch is finished.

declare -a batches=(
    "input_sentences/questions.txt"
    "input_sentences/answers.txt"
    "input_sentences/rationales.txt"
    )

rm .processing_*

for batch in "${batches[@]}"
do
    echo "Processing $batch"
    filename=$(basename $batch)
    echo "$filename"
    touch .processing_$filename
    ./run_parser.sh $batch
    rm .processing_$filename
done