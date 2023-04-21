import sys
import time
import faiss
import numpy as np

if __name__ == '__main__':
    
    dim = 128
    x = int(sys.argv[1])
    k = int(sys.argv[2])
    print('{}M items, K = {}'.format(x, k))
    total_num = x * 1024000

    gpu_res = faiss.StandardGpuResources()

    np.random.seed(1234)
    corpus = np.random.randint(0, 256, (total_num, dim)).astype('float32')
    query  = np.random.randint(0, 256, (1, dim)).astype('float32')

    # index = faiss.IndexLSH(dim, dim)
    measure = faiss.METRIC_INNER_PRODUCT
    param =  'HNSW64' 
    index = faiss.index_factory(dim, param, measure)  
    print(index.is_trained)
    # gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
    # print('convert over')
    index.add(corpus)
    # print('add over')
    start_time = time.time()
    distance, idx = index.search(query, k)
    end_time = time.time()
    run_time = end_time - start_time
    print('Running time : ' + str(run_time) + 's')