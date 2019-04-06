#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream> 
#include <cuda_runtime.h> 
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <intrin.h>
#include <ctime>

#pragma comment(lib, "cudart") 
#define SIZE_M 1024
#define SIZE_N 600000
#define COUNT_OF_THREADS SIZE_M

using namespace std;

void cpu_matrixOperation(short*, short*, int, int);
void cuda_matrixOperation(short*, short*, bool);
void cuda_checkStatus(cudaError_t);
void fillMatrix(short*, int, int);
int getCountOfBlocks(int, int, int);
bool checkEquality(short*, short*, int, int);

int main() {
	auto* initMatrix = (short*)malloc(SIZE_M * SIZE_N * sizeof(short));
	auto* cpu_outMatrix = (short*)malloc(SIZE_M * SIZE_N * sizeof(short));
	auto* cuda_outMatrix = (short*)malloc(SIZE_M * SIZE_N * sizeof(short));
	auto* cuda_outMatrixSharedMemory = (short*)malloc(SIZE_M * SIZE_N * sizeof(short));

	fillMatrix(initMatrix, SIZE_M, SIZE_N);

	/*for (auto i = 0; i < sizeOfN * sizeOfM; i++) {
		cout << inMatrix[i] << " ";
		if ((i + 1) % sizeOfM == 0) {
			cout << endl;
		}
	}*/

	cuda_matrixOperation(initMatrix, cuda_outMatrixSharedMemory, true);
	cuda_matrixOperation(initMatrix, cuda_outMatrix, false);
	cpu_matrixOperation(initMatrix, cpu_outMatrix, SIZE_M, SIZE_N);

	if (checkEquality(cuda_outMatrix, cuda_outMatrixSharedMemory, SIZE_M, SIZE_N)
		&& checkEquality(cpu_outMatrix, cuda_outMatrixSharedMemory, SIZE_M, SIZE_N)) {
		cout << "Results are equals!" << endl;
	}
	else {
		cout << "Results are NOT equals!" << endl;
	}

	/*cout << endl << "Not optimize" << endl;
	for (auto i = 0; i < sizeOfN * sizeOfM; i++) {
		cout << cuda_outMatrix[i] << " ";
		if ((i + 1) % (sizeOfM * 2) == 0) {
			cout << endl;
		}
	}
	cout << endl << "Shared memory" << endl;
	for (auto i = 0; i < sizeOfN * sizeOfM; i++) {
		cout << cuda_outMatrixOptimization[i] << " ";
		if ((i + 1) % (sizeOfM * 2) == 0) {
			cout << endl;
		}
	}
	cout << endl << "CPU" << endl;
	for (auto i = 0; i < sizeOfN * sizeOfM; i++) {
		cout << cpu_outMatrix[i] << " ";
		if ((i + 1) % (sizeOfM * 2) == 0) {
			cout << endl;
		}
	}*/

	free(initMatrix);
	free(cpu_outMatrix);
	free(cuda_outMatrix);
	free(cuda_outMatrixSharedMemory);
}

__global__ void cuda_matrixOperationKernel(short* inMatrix, short* outMatrix) {
	int column = blockIdx.x * COUNT_OF_THREADS + threadIdx.x;

	int elements = ((int*)inMatrix)[column];
	short firstElement = (short)elements;
	short secondElement = (short)(elements >> 16);

	if (threadIdx.x < (COUNT_OF_THREADS / 2)) {
		outMatrix[threadIdx.x * 2 * 2 + blockIdx.x * COUNT_OF_THREADS * 2] = firstElement;
		outMatrix[(threadIdx.x * 2 + 1) * 2 + blockIdx.x * COUNT_OF_THREADS * 2] = secondElement;
	}
	else {
		outMatrix[(threadIdx.x - COUNT_OF_THREADS / 2) * 2 * 2 + 1 + blockIdx.x * COUNT_OF_THREADS * 2] = firstElement;
		outMatrix[((threadIdx.x - COUNT_OF_THREADS / 2) * 2 + 1) * 2 + 1 + blockIdx.x * COUNT_OF_THREADS * 2] = secondElement;
	}
}

__global__ void cuda_matrixOptimizationOperationKernel(short* inMatrix, short* outMatrix) {
	int column = blockIdx.x * COUNT_OF_THREADS + threadIdx.x;

	__shared__ int sharedMemory[COUNT_OF_THREADS];
	__shared__ short sharedMemoryOut[COUNT_OF_THREADS * 2];

	sharedMemory[threadIdx.x] = ((int*)inMatrix)[column];

	int buffer = sharedMemory[threadIdx.x];
	short firstElement = (short)buffer;
	short secondElement = (short)(buffer >> 16);

	if (threadIdx.x < (COUNT_OF_THREADS / 2)) {
		sharedMemoryOut[threadIdx.x * 2 * 2] = firstElement;
		sharedMemoryOut[(threadIdx.x * 2 + 1) * 2] = secondElement;
	}
	else {
		sharedMemoryOut[(threadIdx.x - COUNT_OF_THREADS / 2) * 2 * 2 + 1] = firstElement;
		sharedMemoryOut[((threadIdx.x - COUNT_OF_THREADS / 2) * 2 + 1) * 2 + 1] = secondElement;
	}
	__syncthreads();

	((int*)outMatrix)[threadIdx.x + blockIdx.x * COUNT_OF_THREADS] = ((int*)sharedMemoryOut)[threadIdx.x];
}

void cuda_matrixOperation(short* inMatrix, short* outMatrix, bool optimizationFlag) {
	cudaEvent_t cuda_startTime;
	cudaEvent_t cuda_endTime;
	float resultTime;
	cuda_checkStatus(cudaEventCreate(&cuda_startTime));
	cuda_checkStatus(cudaEventCreate(&cuda_endTime));

	short* device_inMatrix;
	short* device_outMatrix;
	
	for (int i = 0, int times = 1; i < SIZE_N; i += 400000, times++) {
		//for (int j = 0; j < SIZE_M; j += 1024) {
		//	int sizeM, sizeN;

		//	sizeM = (SIZE_M - j) < 1024 ? SIZE_M - j : 1024;
		//	sizeN = (SIZE_N - i) < 400000 ? SIZE_N - i : 400000;

		//	/*if (SIZE_M - j < 1024) {
		//		sizeM = SIZE_M - j;
		//	}
		//	else {
		//		sizeM = 1024;
		//	}*/
		//	/*if (SIZE_N - i < 400000) {
		//		sizeN = sizeN - i;
		//	}
		//	else {
		//		sizeN = 400000;
		//	}*/
		//	cuda_checkStatus(cudaMalloc(&device_inMatrix, sizeM * sizeN * sizeof(short)));
		//	cuda_checkStatus(cudaMalloc(&device_outMatrix, sizeM * sizeN * sizeof(short)));
		//	cuda_checkStatus(cudaMemcpy(device_inMatrix, inMatrix[], sizeM * sizeN * sizeof(short), cudaMemcpyHostToDevice));
		//}
		int sizeN;
		sizeN = (SIZE_N - i) < 400000 ? SIZE_N - i : 400000;
		short b = inMatrix[i * 1024];
		short* a = &inMatrix[i * 1024];
		cuda_checkStatus(cudaMalloc(&device_inMatrix, SIZE_M * sizeN * sizeof(short)));
		cuda_checkStatus(cudaMalloc(&device_outMatrix, SIZE_M  * sizeN * sizeof(short)));
		cuda_checkStatus(cudaMemcpy(device_inMatrix, a, SIZE_M * sizeN * sizeof(short), cudaMemcpyHostToDevice));

		dim3 blockSize(COUNT_OF_THREADS);
		dim3 gridSize(sizeN / 2);

		if (optimizationFlag) {
			cuda_checkStatus(cudaEventRecord(cuda_startTime, NULL));
			cuda_matrixOptimizationOperationKernel << < gridSize, blockSize >> > (device_inMatrix, device_outMatrix);
			cuda_checkStatus(cudaPeekAtLastError());
			cuda_checkStatus(cudaDeviceSynchronize());
			cuda_checkStatus(cudaEventRecord(cuda_endTime, NULL));
			cuda_checkStatus(cudaEventSynchronize(cuda_endTime));
		}
		else {
			cuda_checkStatus(cudaEventRecord(cuda_startTime, NULL));
			cuda_matrixOperationKernel << < gridSize, blockSize >> > (device_inMatrix, device_outMatrix);
			cuda_checkStatus(cudaPeekAtLastError());
			cuda_checkStatus(cudaDeviceSynchronize());
			cuda_checkStatus(cudaEventRecord(cuda_endTime, NULL));
			cuda_checkStatus(cudaEventSynchronize(cuda_endTime));
		}

		cuda_checkStatus(cudaEventElapsedTime(&resultTime, cuda_startTime, cuda_endTime));

		if (optimizationFlag) {
			printf("%d: CUDA time with optimization: %lf seconds\n", times, (double)resultTime / CLOCKS_PER_SEC);
		}
		else {
			printf("%d: CUDA time: %lf seconds\n", times, (double)resultTime / CLOCKS_PER_SEC);
		}

		cuda_checkStatus(cudaMemcpy(&outMatrix[i * 1024], device_outMatrix, SIZE_M * sizeN * sizeof(short), cudaMemcpyDeviceToHost));

		cuda_checkStatus(cudaFree(device_inMatrix));
		cuda_checkStatus(cudaFree(device_outMatrix));
	}
}

void cpu_matrixOperation(short* inMatrix, short* outMatrix, int sizeOfM, int sizeOfN) {
	clock_t startTime, endTime;
	startTime = clock();
	for (auto i = 0; i < sizeOfM; i++) {
		for (auto j = 0; j < sizeOfN; j++) {
			int a = (j + 1) % 2 == 0 ? 1 : 0;
			outMatrix[(j / 2) * sizeOfM * 2 + a + i * 2] = inMatrix[i + sizeOfM * j];
		}
	}
	endTime = clock();
	printf("CPU time: %lf seconds\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);
}


void fillMatrix(short* matrix, int sizeOfM, int sizeOfN) {
	for (int i = 0; i < sizeOfN; i++) {
		for (int j = 0; j < sizeOfM; j++) {
			matrix[sizeOfM * i + j] = rand() % 20 + 1;
		}
	}
}

int getCountOfBlocks(int sizeOfM, int sizeOfN, int countOfThreads) {
	if (sizeOfM * sizeOfN % countOfThreads == 0) {
		return sizeOfM * sizeOfN / countOfThreads;
	}
	else {
		return (sizeOfM * sizeOfN / countOfThreads) + 1;
	}
}

void cuda_checkStatus(cudaError_t cudaStatus) {
	if (cudaStatus != cudaSuccess) {
		cout << "CUDA return error code: " << cudaStatus;
		cout << " " << cudaGetErrorString(cudaStatus) << endl;
		exit(-1);
	}
}

bool checkEquality(short* inMatrix, short* outMatrix, int sizeOfM, int sizeOfN) {
	for (int i = 0; i < sizeOfN * sizeOfM; i++) {
		if (inMatrix[i] != outMatrix[i]) {
			return false;
		}
	}
	return true;
}