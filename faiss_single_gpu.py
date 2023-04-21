import time
import faiss
import numpy as np

if __name__ == '__main__':
    
    dim = 128
    x = int(input('Corpus size = '))
    k = int(input('K = '))
    total_num = x * 1024000

    gpu_res = faiss.StandardGpuResources()

    np.random.seed(1234)
    corpus = np.random.randint(0, 256, (total_num, dim)).astype('float32')
    query  = np.random.randint(0, 256, (1, dim)).astype('float32')

    index = faiss.IndexFlatIP(dim)
    gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, index)

    gpu_index.add(corpus)

    start_time = time.time()
    distance, idx = gpu_index.search(query, k)
    end_time = time.time()
    run_time = end_time - start_time
    print('Running time : ' + str(run_time) + 's')