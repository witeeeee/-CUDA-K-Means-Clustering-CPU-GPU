# CUDA-K-Means-Clustering-CPU-GPU
K-Means clustering algorithm on GPU and CPU with time

**Usage:**

```
nvcc kmeans.cu -o kmeans
./kmeans
```

CPU portion not parallelized

Note: Ratio between N and ThreadsPerBlock cannot increase more than it is by default, am new to CUDA so don't know why. Might be code related, might be device related. For purposes of observing speedup, please try and maintain the ratio or decrease it. 

**References:**

[K-Means using CUDA](http://alexminnaar.com/2019/03/05/cuda-kmeans.html)
[Algorithm](https://www.javatpoint.com/k-means-clustering-algorithm-in-machine-learning)