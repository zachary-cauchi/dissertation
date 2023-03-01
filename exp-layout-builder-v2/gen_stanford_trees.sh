#!/usr/bin/env bash
#
# Runs the parser on the given file, outputting results in the format we need.

print_usage () {
  echo Usage: `basename $0` 'filename prefix'
  echo Example: `basename $0` question_answers
  echo
}

# Check that we received an input filename prefix such as 'question_answers'
if [ ! $# -ge 1 ]; then
  print_usage
  exit
fi

scriptdir=`dirname $0`
filename=`basename $*`
outputdir=parser-trees/
log_file=out.txt

# Get the max amount of processors we're comfortable using.
max_proc_count=$(nproc --ignore=1)

# Find all the input files matching our given prefix.
sentence_files=($(find input-sentences/ -type f -iname "$filename.*.txt"))

# Make sure there were files found matching the prefix. If none were found, exit.
if [ ${#sentence_files[@]} -eq 0 ]; then
    echo "No files found for $filename"
    print_usage
    exit
else
    echo "Processing ${#sentence_files[@]} files."
fi

# Sort them in order, for pretty-printing.
IFS=$'\n' sentence_files=($(sort <<<"${sentence_files[*]}"))
unset IFS

# Main entrypoint for each file to parse.
run_for_each () {
  local each=$1
  echo "Processing: $each"
  scriptdir=`dirname $0`
  filename=`basename $*`

  # Execute the parser on its name.
  java -mx4000m -cp "$scriptdir/*:$PWD/stanford-parser/*:" edu.stanford.nlp.parser.lexparser.LexicalizedParser \
  -outputFormat "oneline" \
  -outputFormatOptions "stem,collapsedDependencies,includeTags" \
  -sentences newline \
  edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz $* | tee parser-trees/$filename
}

export -f run_for_each

# Execute the parser on all the found files using GNU Parallels (Citation needed.)
parallel -j $max_proc_count --line-buffer --bar --tagstring '{/} <- ' run_for_each {} ::: "${sentence_files[@]}"
