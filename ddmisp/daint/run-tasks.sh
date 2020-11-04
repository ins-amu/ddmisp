#!/bin/bash


BATCH_FILE=$1
LINES_FROM=$2
LINES_TO=$3

NLINES=$(cat $BATCH_FILE | wc -l)

for i in $(seq $LINES_FROM $LINES_TO); do
    if [ "$i" -ge "1" ] && [ "$i" -le "$NLINES" ]; then
        eval $(sed "${i}q;d" $BATCH_FILE) &
    fi
done

wait
