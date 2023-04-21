#!/bin/bash

echo "Faiss_CPU test result:"
for i in {1..9}
do
    python ./faiss_cpu.py $i 1024
done

echo "Faiss_single_GPU test result:"
for i in {1..9}
do
    python ./faiss_single_gpu.py $i 1024
done

echo "Faiss_multi_GPU test result:"
for i in {1..9}
do
    python ./faiss_multi_gpu.py $i 1024
done