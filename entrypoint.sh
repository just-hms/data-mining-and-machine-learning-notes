#!/bin/sh

mkdir build -p

# create un bundle.md

cat $(ls docs/[0-9]-*) $(ls docs/[0-9][0-9]-*) > build/bundle.md

# convert to html


pandoc --toc --standalone  --mathjax -f markdown -t html build/bundle.md -o build/bundle.html --verbose --template=./template.html --metadata title="dm-ml"
# 


# inject latex?

# convrt to pdf

