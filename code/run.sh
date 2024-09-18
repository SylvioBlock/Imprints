#! /bin/bash

cnn_model=inference.h5 
experiment=S04__21779169__20170702_124357104
experiment_path=/path/dataset/${experiment}
out_path=output
show=True

rm ${out_path};

mkdir ${out_path};

python3 main.py \
	${experiment_path} \
	${cnn_model} \
	${experiment} \
	${out_path} \
	${show}

