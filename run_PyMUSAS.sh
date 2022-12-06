#!/bin/bash
cat img/data.txt | sudo docker run -i --rm ghcr.io/ucrel/cytag:1.0.4 > img/welsh_text_example.tsv
