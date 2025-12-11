#!/usr/bin/env bash
# Reassemble split parts for COVID19_twitter_full_dataset.csv.gz
cat COVID19_twitter_full_dataset.csv.gz.part-* > COVID19_twitter_full_dataset.csv.gz
