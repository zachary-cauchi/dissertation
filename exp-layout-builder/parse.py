#!/usr/bin/env python2

# Sourced from: https://gist.github.com/ronghanghu/67aeb391f4839611d119c73eba53bc5f
# This script is used to generate our parsing on the VQA dataset in
# https://github.com/ronghanghu/n2nmn/tree/master/exp_vqa/data/parse/new_parse
#
# Run this script on the output from Stanford parser with the following command:
#
# java -mx150m -cp "$scriptdir/*:" edu.stanford.nlp.parser.lexparser.LexicalizedParser \
#  -outputFormat "words,typedDependencies" -outputFormatOptions "stem,collapsedDependencies,includeTags" \
#  -sentences newline \
#  edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz \
#  $*
#
# The Stanford parser should output a parsing tree for each question (e.g. "are there patients?") like the line below
#   (ROOT (SQ (VBP are) (NP (EX there)) (ADJP (VBG patients)) (. ?)))
# which is used as input to parse.py

from nltk.tree import Tree, ParentedTree
import sys
import re

KEEP = [
    ("WHNP", "WH"),
    ("WHADVP", "WH"),
    (r"NP", "NP"),
    ("VP", "VP"),
    ("PP", "PP"),
    ("ADVP", "AP"),
    ("ADJP", "AP"),
    ("this", "null"),
    ("these", "null"),
    ("it", "null"),
    ("EX", "null"),
    ("PRP$", "null"),
]
KEEP = [(re.compile(k), v) for k, v in KEEP]

def flatten(tree):
    if not isinstance(tree, list):
        return [tree]
    return sum([flatten(s) for s in tree], [])

def collect_span(term):
    parts = flatten(term)
    lo = 1000
    hi = -1000
    for part in parts:
        assert isinstance(part, tuple) and len(part) == 2
        lo = min(lo, part[1][0])
        hi = max(hi, part[1][1])
    assert lo < 1000
    assert hi > -1000
    return (lo, hi)

def finalize(col, top=True):
    dcol = despan(col)
    is_wh = isinstance(dcol, list) and len(dcol) > 1 and flatten(dcol[0])[0] == "WH"
    out = []
    if not top:
        rest = col
    elif is_wh:
        whspan = flatten(col[0])[0][1]
        out.append("describe[%s,%s]" % (whspan))
        rest = col[1:]
    else:
        out.append("is")
        rest = col
    if len(rest) == 0:
        return out
    elif len(rest) == 1:
        body = out
    else:
        body = ["and"]
        out.append(body)
    for term in rest:
        if term[0][0] == "PP":
            span_below = collect_span(term[1:])
            span_full = term[0][1]
            span_here = (span_full[0], span_below[0])
            body.append(["relate[%s,%s]" % span_here, finalize(term[1:], top=False)])
        elif isinstance(term, tuple) and isinstance(term[0], str):
            body.append("find[%s,%s]" % term[1])
        else:
            # TODO more structure here
            body.append("find[%s,%s]" % collect_span(term))
    if len(body) > 3:
        del body[3:]
    if isinstance(out, list) and len(out) == 1:
        out = out[0]
    return out

def strip(tree):
    if not isinstance(tree, Tree):
        label = tree
        flat_children = []
        span = ()
    else:
        label = tree.label()
        children = [strip(child) for child in next(tree.subtrees())]
        flat_children = sum(children, [])
        leaves = tree.leaves()
        span = (int(leaves[0]), int(leaves[-1]) + 1)

    proj_label = [v for m, v in KEEP if m.match(label)]
    if len(proj_label) == 0:
        return flat_children
    else:
        return [[(proj_label[0], span)] + flat_children]

def despan(rr):
    out = []
    for r in rr:
        if isinstance(r, tuple) and len(r) == 2 and isinstance(r[1], tuple):
            out.append(r[0])
        elif isinstance(r, list):
            out.append(despan(r))
        else:
            out.append(r)
    return out

def collapse(tree):
    if not isinstance(tree, list):
        return tree
    rr = [collapse(st) for st in tree]
    rr = [r for r in rr if r != []]
    drr = despan(rr)
    if drr == ["NP", ["null"]]:
        return []
    if drr == ["null"]:
        return []
    if drr == ["PP"]:
        return []
    members = set(flatten(rr))
    if len(members) == 1:
        return list(members)
    if len(drr) == 2 and drr[0] == "VP" and isinstance(drr[1], list):
        if len(drr[1]) == 0:
            return []
        elif drr[1][0] == "VP" and len(drr[1]) == 2:
            return [rr[1][0], rr[1][1]]
    return rr

def pp(lol):
    if isinstance(lol, str):
        return lol
    return "(%s)" % " ".join([pp(l) for l in lol])

with open(sys.argv[1]) as ptb_f:
    for line in ptb_f:
        tree = ParentedTree.fromstring(line)
        index = 0
        for st in tree.subtrees():
            if len(list(st.subtrees())) == 1:
                st[0] = str(index)
                index += 1
        colparse = collapse(strip(tree))
        final = finalize(colparse)
        print(pp(final))
        #print pp(final)
        #print " ".join(tree.leaves())
        #print colparse
        #print finalize(colparse)
        #print
