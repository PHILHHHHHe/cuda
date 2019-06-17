#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#include "curand_kernel.h" 

#include <stdio.h>
#include <time.h>


__global__ void reduce0(int *g_idata, int *g_odata) {
	extern __shared__ int sdata[];

	//one element from global to share
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();

	//reduction
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2 * s) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
		
	}

	//write result
	if (tid == 0) {
		g_odata[blockIdx.x] = sdata[0];
		//printf("r = %d\n",sdata[0]);
	}
	
}

__global__ void reduce1(int *g_idata, int *g_odata) {
	extern __shared__ int sdata[];

	//one element from global to share
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();

	//reduction
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		int index = 2 * s * tid;
		if (index < blockDim.x) {
			sdata[index] += sdata[index + s];
		}
		__syncthreads();
	}
	//write result
	if (tid == 0) {
		g_odata[blockIdx.x] = sdata[0];
		printf("r = %d\n", sdata[0]);
	}
}


__global__ void reduce2(int *g_idata, int *g_odata) {
	extern __shared__ int sdata[];

	//one element from global to share
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();

	//reduction
	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1) {
		if (tid <s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	//write result
	if (tid == 0) {
		g_odata[blockIdx.x] = sdata[0];
		//printf("r = %d\n", sdata[0]);
	}

}


__global__ void reduce3(int *g_idata, int *g_odata) {
	extern __shared__ int sdata[];

	//one element from global to share
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x*2) + threadIdx.x;
	sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
	__syncthreads();

	//reduction
	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1) {
		if (tid <s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	//write result
	if (tid == 0) {
		g_odata[blockIdx.x] = sdata[0];
		//printf("r = %d\n", sdata[0]);
	}

}

__global__ void reduce4(int *g_idata, int *g_odata) {
	extern __shared__ int sdata[];

	//one element from global to share
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
	__syncthreads();

	//reduction
	for (unsigned int s = blockDim.x / 2; s>32; s >>= 1) {
		if (tid <s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	//write result
	if (tid < 32) {
		sdata[tid] += sdata[tid + 32];
		sdata[tid] += sdata[tid + 16];
		sdata[tid] += sdata[tid + 8];
		sdata[tid] += sdata[tid + 4];
		sdata[tid] += sdata[tid + 2];
		sdata[tid] += sdata[tid + 1];
	}
	//if (tid == 0) printf("result = %d", sdata[0]);
}

template <unsigned int blockSize>
__global__ void reduce5(int *g_idata, int *g_odata) {
	extern __shared__ int sdata[];

	//one element from global to share
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
	__syncthreads();

	//reduction
	if (blockSize >= 128) {
		if (tid < 64) {
			sdata[tid] += sdata[tid + 128];
		} __syncthreads();
	}

	
	//write result
	if (tid < 32) {
		if(blockSize >= 64) sdata[tid] += sdata[tid + 32];
		if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
		if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
		if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
		if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
		if (blockSize >= 3) sdata[tid] += sdata[tid + 1];
	}
	//if (tid == 0) printf("result = %d", sdata[0]);
}


int main()
{
	int h_a[1536];
	int sum = 0;
	for (int i = 1; i <= 1536; i++) {
		h_a[i - 1] = i;
		//sum += i;
	}
	
	int *d_a, *d_out, *bbb;

	cudaMalloc((void**)&d_a, 1536 * sizeof(int));
	cudaMalloc((void**)&d_out, 1536 * sizeof(int));
	cudaMemcpy(d_a, h_a, 1536 * sizeof(int),cudaMemcpyHostToDevice);

	cudaStream_t stream;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	reduce0 <<< 6, 256, 1536 >>> (d_a, d_out);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf( "\n reduce0总时间(double) %f ms.\n", elapsedTime);

	
	cudaEventRecord(start, 0);
	reduce1 << < 6, 256, 1536 >> > (d_a, d_out);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	//float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("\n reduce1总时间(double) %f ms.\n", elapsedTime);

	cudaEventRecord(start, 0);
	reduce2 << < 6, 256, 1536 >> > (d_a, d_out);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	//float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("\n reduce2总时间(double) %f ms.\n", elapsedTime);

	cudaEventRecord(start, 0);
	reduce3 << < 3, 256, 1536 >> > (d_a, d_out);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	//float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("\n reduce3总时间(double) %f ms.\n", elapsedTime);

	cudaEventRecord(start, 0);
	reduce4 << < 3, 256, 1536 >> > (d_a, d_out);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	//float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("\n reduce4总时间(double) %f ms.\n", elapsedTime);
	cudaStreamCreate(&stream);
	cudaEventRecord(start, 0);
	reduce5<64> << < 3, 256, 1536, stream >> > (d_a, d_out);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	//float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("\n reduce5总时间(double) %f ms.\n", elapsedTime);

	cudaStreamDestroy(stream);
	cudaFree(d_a);
	cudaFree(d_out);

	return 0;
	
}
