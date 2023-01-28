#!/bin/sh

# create un bundle.md
files=$(ls docs/[0-9]-* ; ls docs/[0-9][0-9]-*)

echo "" > bundle.md

for file in $files; do
    cat "$file" >> bundle.md
    echo "\n\n" >> bundle.md
done

# convert to html
pandoc \
    --toc --standalone --verbose --mathjax \
    -f markdown -t html bundle.md \
    -o bundle.html --template=./template.html \
    --metadata title="Data mining and Machine Learning" \
    --number-sections --toc-depth=3 \
    --lua-filter=adjust_number_depth.lua

rm bundle.md

