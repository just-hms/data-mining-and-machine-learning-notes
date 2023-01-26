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
pandoc --toc --standalone -f markdown -t html build/bundle.md -o build/bundle.html --verbose --template=./template.html --metadata title="dm-ml" --mathjax --number-sections

# convert to pdf

# pandoc -s build/bundle.html -o build/output.pdf --mathjax
