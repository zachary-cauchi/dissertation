#!/usr/bin/env bash
#
# Runs the parser on the given file, outputting results in the format we need.

if [ ! $# -ge 1 ]; then
  echo Usage: `basename $0` 'file(s)'
  echo
  exit
fi

scriptdir=`dirname $0`
filename=`basename $*`

java -mx8000m -cp "$scriptdir/*:$PWD/stanford-parser/*:" edu.stanford.nlp.parser.lexparser.LexicalizedParser \
 -outputFormat "oneline" \
 -outputFormatOptions "stem,collapsedDependencies,includeTags" \
 -sentences newline \
 edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz $* | tee parsed/$filename
 