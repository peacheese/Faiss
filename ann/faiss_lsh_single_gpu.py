import sys
import time
import faiss
import numpy as np

if __name__ == '__main__':
    
    dim = 128
    x = int(sys.argv[1])
    k = int(sys.argv[2])
    print('{}M items, Top-K = {}'.format(x, k))
    total_num = x * 1024000

    gpu_res = faiss.StandardGpuResources()

    # np.random.seed(1234)
    corpus = np.random.randint(0, 256, (total_num, dim)).astype('float32')
    query  = np.random.randint(0, 256, (1, dim)).astype('float32')

    index = faiss.IndexLSH(dim, dim * 2)
    gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, index)

    gpu_index.add(corpus)

    start_time = time.time()
    distance, idx = gpu_index.search(query, k)
    end_time = time.time()
    run_time = end_time - start_time
    print('Running time : ' + str(run_time) + 's')

    acc_index = faiss.IndexFlatIP(dim)
    gpu_acc_index = faiss.index_cpu_to_gpu(gpu_res, 0, acc_index)
    gpu_acc_index.add(corpus)
    distance, acc_idx = gpu_acc_index.search(query, k)
    
    count = 0
    for item in idx[0]:
        if item in acc_idx[0]:
            count += 1
    print(count)