#!/bin/sh

mkdir build -p

# create un bundle.md

cat $(ls docs/[0-9]-*) $(ls docs/[0-9][0-9]-*) > build/bundle.md

{ echo -n '<!-- end-of-table-of-content -->\n\n'; cat build/bundle.md; } > build/tmp.md
mv build/tmp.md build/bundle.md

# creiamo la toc
./toc.sh build/bundle.md

# convertiamo a html

pandoc build/bundle.md -s -o build/bundle.html --mathjax --metadata title="DM-ML notes"

# injectiamo il latex

# convrtiamo a pdf

