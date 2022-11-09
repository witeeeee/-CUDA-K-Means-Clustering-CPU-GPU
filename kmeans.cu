#include <stdio.h>
#include <time.h>

#define N 4096
#define ThreadsPerBlock 256
#define K 3
#define MAX_ITER 10000

__device__ float distance(float x1, float x2)
{
	return sqrt((x2-x1)*(x2-x1));
}

float host_distance(float x1, float x2) 
{
    return sqrt((x2-x1)*(x2-x1));
}

void CPUkMeansClusterAssignment(float *h_datapoints, int *h_clust_assn, float *h_centroids)
{
    for(int i = 0; i<N; i++) 
    {
        float min_dist = INFINITY;
        int closest_centroid = 0;

        for(int c = 0; c<K; ++c)
        {
            float dist = host_distance(h_datapoints[i], h_centroids[c]);
            if(dist < min_dist)
            {
                min_dist = dist;
                closest_centroid=c;
            }
        }
        h_clust_assn[i]=closest_centroid;
    }
}

void CPUkMeansCentroidUpdate(float *h_datapoints, int *h_clust_assn, float *h_centroids, int *h_clust_sizes) 
{
    for(int i = 0; i<N; i++) 
    {
        int clust_id = h_clust_assn[i];
        h_centroids[clust_id] += h_datapoints[i];
        h_clust_sizes[clust_id] += 1;
    }
    
    for(int z = 0; z<K; z++)
        h_centroids[z] = h_centroids[z] / h_clust_sizes[z];
}

__global__ void kMeansClusterAssignment(float *dev_datapoints, int *dev_clust_assn, float *dev_centroids)
{
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx >= N) 
        return;

	//finding the closest centroid to this datapoint
	float min_dist = INFINITY;
	int closest_centroid = 0;

	for(int c = 0; c<K; ++c)
	{
		float dist = distance(dev_datapoints[idx],dev_centroids[c]);
		if(dist < min_dist)
		{
			min_dist = dist;
			closest_centroid=c;
		}
	}

	//assign closest cluster id for this datapoint/thread
	dev_clust_assn[idx]=closest_centroid;
}

__global__ void kMeansCentroidUpdate(float *dev_datapoints, int *dev_clust_assn, float *dev_centroids, int *dev_clust_sizes)
{
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx >= N) 
        return;

	//get idx of thread at the block level
	const int s_idx = threadIdx.x;

	//put the datapoints and corresponding cluster assignments in shared memory so that they can be summed by thread 0 later
	__shared__ float s_datapoints[ThreadsPerBlock];
	s_datapoints[s_idx]= dev_datapoints[idx];

	__shared__ int s_clust_assn[ThreadsPerBlock];
	s_clust_assn[s_idx] = dev_clust_assn[idx];

	__syncthreads();

	//it is the thread with idx 0 (in each block) that sums up all the values within the shared array for the block it is in
	if(s_idx==0)
	{
		float b_clust_datapoint_sums[K]={0, 0, 0};
		int b_clust_sizes[K]={0, 0, 0};

		for(int j=0; j<blockDim.x; ++j)
		{
			int clust_id = s_clust_assn[j];
			b_clust_datapoint_sums[clust_id]+=s_datapoints[j];
			b_clust_sizes[clust_id]+=1;
		}

		//Now we add the sums to the global centroids and add the counts to the global counts.
		for(int z=0; z < K; ++z)
		{
			atomicAdd(&dev_centroids[z],b_clust_datapoint_sums[z]);
			atomicAdd(&dev_clust_sizes[z],b_clust_sizes[z]);
		}
	}

	__syncthreads();

	//currently centroids are just sums, so divide by size to get actual centroids 
	if(idx < K){
		dev_centroids[idx] = dev_centroids[idx]/dev_clust_sizes[idx]; 
	}
}

int main()
{
	float *dev_datapoints=0;
	int *dev_clust_assn = 0;
	float *dev_centroids = 0;
	int *dev_clust_sizes=0;
    cudaEvent_t start, stop;     	
	float elapsed_time_ms; 

	cudaMalloc(&dev_datapoints, N*sizeof(float));
	cudaMalloc(&dev_clust_assn,N*sizeof(int));
	cudaMalloc(&dev_centroids,K*sizeof(float));
	cudaMalloc(&dev_clust_sizes,K*sizeof(float));

	float *host_centroids = (float*)malloc(K*sizeof(float));
	float *host_datapoints = (float*)malloc(N*sizeof(float));
	int *host_clust_sizes = (int*)malloc(K*sizeof(int));
    int *host_clust_assn = (int*)malloc(N*sizeof(int));

    float *Chost_centroids = (float*)malloc(K*sizeof(float));
	float *Chost_datapoints = (float*)malloc(N*sizeof(float));
	int *Chost_clust_sizes = (int*)malloc(K*sizeof(int));
    int *Chost_clust_assn = (int*)malloc(N*sizeof(int));

	srand(time(0));

	for(int c=0;c<K;++c)
	{
		host_centroids[c]=(float) rand() / (double)RAND_MAX; //initializing random centroids
        Chost_centroids[c] = host_centroids[c];
		printf("%f\n", host_centroids[c]);
		host_clust_sizes[c]=0;
        Chost_clust_sizes[c]=0;
	}

	//initalize datapoints
	for(int d = 0; d < N; ++d)
	{
		host_datapoints[d] = (float) rand() / (double)RAND_MAX;
        Chost_datapoints[d] = host_datapoints[d];
	}

	cudaMemcpy(dev_centroids,host_centroids,K*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_datapoints,host_datapoints,N*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_clust_sizes,host_clust_sizes,K*sizeof(int),cudaMemcpyHostToDevice);

	int cur_iter = 1;

    cudaEventCreate(&start);     	
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

	while(cur_iter < MAX_ITER)
	{
		//call cluster assignment kernel
		kMeansClusterAssignment<<<(N+ThreadsPerBlock-1)/ThreadsPerBlock,ThreadsPerBlock>>>(dev_datapoints,dev_clust_assn,dev_centroids);

		//copy new centroids back to host 
		cudaMemcpy(host_centroids,dev_centroids,K*sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(host_clust_assn,dev_clust_assn,K*sizeof(float),cudaMemcpyDeviceToHost);

         if(cur_iter == MAX_ITER-1) //printing the GPU calculated centroids at the last iteration
            for(int i =0; i < K; ++i){
                printf("Iteration %d: centroid %d: %f\n",cur_iter,i,host_centroids[i]);
            }

		//reset centroids and cluster sizes
		cudaMemset(dev_centroids,0.0,K*sizeof(float));
		cudaMemset(dev_clust_sizes,0,K*sizeof(int));

		//call centroid update kernel
		kMeansCentroidUpdate<<<(N+ThreadsPerBlock-1)/ThreadsPerBlock,ThreadsPerBlock>>>(dev_datapoints,dev_clust_assn,dev_centroids,dev_clust_sizes);

		cur_iter+=1;
	}

    cudaEventRecord(stop, 0);     
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_ms, start, stop );

    printf("Time to calculate results on GPU: %f ms.\n", elapsed_time_ms); 

    cur_iter = 1;

    cudaEventRecord(start, 0);


    while(cur_iter < MAX_ITER)
	{
		//call cluster assignment kernel
        CPUkMeansClusterAssignment(host_datapoints,Chost_clust_assn, Chost_centroids);
		
        if(cur_iter == MAX_ITER-1)  //printing the CPU calculated centroids at the last iteration.
            for(int i =0; i < K; ++i){
                printf("Iteration %d: centroid %d: %f\n",cur_iter,i,Chost_centroids[i]);
            }

        memset(Chost_centroids, 0.0, K*sizeof(float));
        memset(Chost_clust_sizes, 0, K*sizeof(int));

		CPUkMeansCentroidUpdate(Chost_datapoints,Chost_clust_assn,Chost_centroids,Chost_clust_sizes);

		cur_iter+=1;
	}

    cudaEventRecord(stop, 0);     	
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_ms, start, stop );

    printf("Time to calculate results on CPU: %f ms.\n", elapsed_time_ms); 

	cudaFree(dev_datapoints);
	cudaFree(dev_clust_assn);
	cudaFree(dev_centroids);
	cudaFree(dev_clust_sizes);

	free(host_centroids);
	free(host_datapoints);
	free(host_clust_sizes);
    free(host_clust_assn);

    free(Chost_centroids);
	free(Chost_datapoints);
	free(Chost_clust_sizes);
    free(Chost_clust_assn);

	return 0;
}