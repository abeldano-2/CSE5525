#!/usr/bin/env bash
# Reassemble split parts for full_dataset.tsv.gz
cat full_dataset.tsv.gz.part-* > full_dataset.tsv.gz

