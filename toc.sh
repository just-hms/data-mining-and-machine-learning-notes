#!/usr/bin/env bash

## Script to generate and add to the Top of a markdown file a table of content
## It uses one parameter that is the name of the markdown file. If none is given, it will look for "README.md"

## Get the name of the file to create the table of content (default is README.md)
file="${1:-README.md}"

## *unique* pattern to look into the file to generate the table of content
## Insert the next HTML line at the beginning of the file (i.e. where the Table of Content will end)
## <!-- end-of-table-of-content -->

## Generate new Table of Content
grep -E "^#{1,5} " $file | sed -E 's/(#+) (.+)/\1:\2:\2/g' | awk -F ":" '{ gsub(/#/,"  ",$1); gsub(/[ ]/,"-",$3); print $1 "- [" $2 "](#" tolower($3) ")" }' > _table.txt

## Insert the new table in the file
sed -i '1 r _table.txt' $file

## clean temporary file
rm _table.txt