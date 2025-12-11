#!/usr/bin/env bash
# Reassemble split parts for tweetid_userid_keyword_sentiments_emotions.csv.gz
cat tweetid_userid_keyword_sentiments_emotions.csv.gz.part-* > tweetid_userid_keyword_sentiments_emotions.csv.gz
