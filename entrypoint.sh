#!/bin/sh

mkdir build -p

# create un bundle.md
files=$(ls docs/[0-9]-* ; ls docs/[0-9][0-9]-*)

echo "" > build/bundle.md

for file in $files; do
    cat "$file" >> build/bundle.md
    echo "\n\n" >> build/bundle.md
done

# convert to html
pandoc \
    --toc --standalone --verbose --mathjax \
    -f markdown -t html build/bundle.md \
    -o build/index.html --template=./template.html \
    --metadata title="Data mining and Machine Learning" \
    --number-sections --toc-depth=3 \
    --lua-filter=adjust_number_depth.lua

rm build/bundle.md
