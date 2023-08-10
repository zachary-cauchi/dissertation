#!/bin/bash
set -e

CORPUS=corpus_vcr.txt
VOCAB_FILE=vocabulary_vcr.txt
COOCCURRENCE_FILE=cooccurrence.bin
COOCCURRENCE_SHUF_FILE=cooccurrence.shuf.bin
BUILDDIR=../../../GloVe/build
SAVE_FILE=vocabulary_vcr_glove
VERBOSE=2
MEMORY=16.0
VOCAB_MIN_COUNT=5
VECTOR_SIZE=300
MAX_ITER=30
WINDOW_SIZE=15
BINARY=2
NUM_THREADS=8
X_MAX=10
if hash python 2>/dev/null; then
    PYTHON=python
else
    PYTHON=python3
fi

# Convert program path to absolute path
BUILDDIR=`realpath $BUILDDIR`

echo
echo "$ $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE"
$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE

echo "$ $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE

echo "$ $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE"
$BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE

echo "$ cut -f 2- -d ' ' $SAVE_FILE.txt > new_$SAVE_FILE.txt"
cut -f 2- -d ' ' $SAVE_FILE.txt > trimmed_$SAVE_FILE.txt

echo "$ cp -f $SAVE_FILE.txt trimmed_$SAVE_FILE.txt"
cp -f trimmed_$SAVE_FILE.txt $SAVE_FILE.txt

echo "$ $PYTHON -c \"import numpy as np; np.save('$SAVE_FILE.npy', np.loadtxt('$SAVE_FILE.txt'), allow_pickle=True)\""
$PYTHON -c "import numpy as np; np.save('$SAVE_FILE.npy', np.loadtxt('$SAVE_FILE.txt'), allow_pickle=True)"