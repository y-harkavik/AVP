#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_image.h"

#include <iostream> 
#include <cuda_runtime.h> 
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <intrin.h>
#include <windows.h>
#include <ctime>
#include <cmath>
#include <cstdlib>

#pragma comment(lib, "cudart") 

#define BLOCK_SIZE_X 1024
#define BLOCK_SIZE_Y 1
#define IMAGE_WIDTH 17374
#define IMAGE_HEIGHT 27472
//#define IMAGE_WIDTH 6000
//#define IMAGE_HEIGHT 4000
#define MAX_BLOCKS 200000

using namespace std;

void cpu_filterImage(BYTE*, BYTE*, int, int);
void cuda_filterImage(BYTE*, BYTE*, bool);
void cuda_checkStatus(cudaError_t);
void resizeImage(BYTE*, BYTE*, int, int);
bool checkEquality(BYTE*, BYTE*, int, int);

int main() {
	unsigned int imageWidth = 0, imageHeight = 0, channels;

	const char primaryImagePath[] = "D:\\big.pgm";
	const char outputImageCpuPath[] = "D:\\imageCPU.pgm";
	const char outputImageGpuPath[] = "D:\\imageGPU.pgm";
	const char outputImageGpuSharedPath[] = "D:\\imageGPUshared.pgm";

	BYTE *primaryImage = NULL;

	__loadPPM(primaryImagePath, &primaryImage, &imageWidth, &imageHeight, &channels);

	auto* outputImageCpu = (BYTE*)malloc(imageWidth * imageHeight * sizeof(BYTE));
	auto* outputImageGpu = (BYTE*)malloc(imageWidth * imageHeight * sizeof(BYTE));
	auto* outputImageGpuShared = (BYTE*)malloc(imageWidth * imageHeight * sizeof(BYTE));
	auto* resizedImage = (BYTE*)malloc((imageWidth + 2) * (imageHeight + 2) * sizeof(BYTE));

	resizeImage(primaryImage, resizedImage, imageWidth, imageHeight);

	cuda_filterImage(resizedImage, outputImageGpuShared, true);
	cuda_filterImage(resizedImage, outputImageGpu, false);
	cpu_filterImage(resizedImage, outputImageCpu, imageWidth, imageHeight);

	__savePPM(outputImageCpuPath, outputImageCpu, imageWidth, imageHeight, channels);
	__savePPM(outputImageGpuPath, outputImageGpu, imageWidth, imageHeight, channels);
	__savePPM(outputImageGpuSharedPath, outputImageGpuShared, imageWidth, imageHeight, channels);

	if (checkEquality(outputImageCpu, outputImageGpu, IMAGE_WIDTH, IMAGE_HEIGHT) &&
		checkEquality(outputImageGpu, outputImageGpuShared, IMAGE_WIDTH, IMAGE_HEIGHT)) {
		cout << "Results are equals!" << endl;
	}
	else {
		cout << "Results are NOT equals!" << endl;
	}

	free(primaryImage);
	free(resizedImage);
	free(outputImageGpu);
	free(outputImageCpu);
}

__device__ WORD pack(uchar4 pixelLine)
{
	return (pixelLine.y << 8) | pixelLine.x;
}

__device__ uchar2 unpack(WORD c)
{
	uchar2 pixelLine;
	pixelLine.x = (BYTE)(c & 0xFF);
	pixelLine.y = (BYTE)((c >> 8) & 0xFF);

	return pixelLine;
}

__device__ BYTE computeFirstPixel(
	uchar2 firstRowFirstTwoPixels,
	uchar2 firstRowSecondTwoPixels,
	uchar2 secondRowFirstTwoPixels,
	uchar2 secondRowSecondTwoPixels,
	uchar2 thirdRowFirstTwoPixels,
	uchar2 thirdRowSecondTwoPixels)
{
	uint32_t result = 0;

	result += firstRowFirstTwoPixels.x;
	result += firstRowFirstTwoPixels.y;
	result += firstRowSecondTwoPixels.x;

	result += secondRowFirstTwoPixels.x;
	result += secondRowFirstTwoPixels.y;
	result += secondRowSecondTwoPixels.x;

	result += thirdRowFirstTwoPixels.x;
	result += thirdRowFirstTwoPixels.y;
	result += thirdRowSecondTwoPixels.x;

	result /= 9;

	if (result > 255) {
		result = 255;
	}

	return (BYTE)result;
}

__device__ BYTE computeSecondPixel(
	uchar2 firstRowFirstTwoPixels,
	uchar2 firstRowSecondTwoPixels,
	uchar2 secondRowFirstTwoPixels,
	uchar2 secondRowSecondTwoPixels,
	uchar2 thirdRowFirstTwoPixels,
	uchar2 thirdRowSecondTwoPixels)
{
	uint32_t result = 0;

	result += firstRowFirstTwoPixels.y;
	result += firstRowSecondTwoPixels.x;
	result += firstRowSecondTwoPixels.y;

	result += secondRowFirstTwoPixels.y;
	result += secondRowSecondTwoPixels.x;
	result += secondRowSecondTwoPixels.y;

	result += thirdRowFirstTwoPixels.y;
	result += thirdRowSecondTwoPixels.x;
	result += thirdRowSecondTwoPixels.y;

	result /= 9;

	if (result > 255) {
		result = 255;
	}

	return (BYTE)result;
}

__global__ void cuda_matrixOperationKernel(BYTE* inMatrix, BYTE* outMatrix, size_t pitchInMatrix, size_t pitchOutMatrix) {
	int remainderElements = (IMAGE_WIDTH % (blockDim.x * 2)) / 2;

	if (remainderElements != 0 && (blockIdx.x + 1) % gridDim.x == 0 && threadIdx.x >= remainderElements) {
		return;
	}

	BYTE *startOfProcessingRow = &inMatrix[pitchInMatrix * blockIdx.y + blockIdx.x * blockDim.x * 2];

	WORD a1 = *((WORD*)&startOfProcessingRow[threadIdx.x * 2]);
	WORD a2 = *((WORD*)&startOfProcessingRow[threadIdx.x * 2 + 2]);

	WORD b1 = *((WORD*)&startOfProcessingRow[threadIdx.x * 2 + pitchInMatrix]);
	WORD b2 = *((WORD*)&startOfProcessingRow[threadIdx.x * 2 + pitchInMatrix + 2]);

	WORD c1 = *((WORD*)&startOfProcessingRow[threadIdx.x * 2 + pitchInMatrix * 2]);
	WORD c2 = *((WORD*)&startOfProcessingRow[threadIdx.x * 2 + pitchInMatrix * 2 + 2]);

	uchar2 aa1 = unpack(a1);
	uchar2 aa2 = unpack(a2);

	uchar2 bb1 = unpack(b1);
	uchar2 bb2 = unpack(b2);

	uchar2 cc1 = unpack(c1);
	uchar2 cc2 = unpack(c2);

	BYTE res1 = computeFirstPixel(aa1, aa2, bb1, bb2, cc1, cc2);
	BYTE res2 = computeSecondPixel(aa1, aa2, bb1, bb2, cc1, cc2);

	outMatrix[blockIdx.y * pitchOutMatrix + threadIdx.x * 2 + blockIdx.x * blockDim.x * 2] = res1;
	outMatrix[blockIdx.y * pitchOutMatrix + threadIdx.x * 2 + 1 + blockIdx.x * blockDim.x * 2] = res2;
}

__global__ void cuda_matrixSharedMemoryOperationKernel(BYTE* inMatrix, BYTE* outMatrix, size_t pitchInMatrix, size_t pitchOutMatrix) {
	int remainderElements = (IMAGE_WIDTH % (blockDim.x * 2)) / 2;

	if (remainderElements != 0 && (blockIdx.x + 1) % gridDim.x == 0 && threadIdx.x >= remainderElements) {
		return;
	}

	__shared__ WORD sharedMemory[3][BLOCK_SIZE_X + 1];
	__shared__ WORD sharedMemoryOut[BLOCK_SIZE_X];

	WORD* startOfProcessingRow = (WORD*)&inMatrix[blockIdx.y * pitchInMatrix + blockIdx.x * blockDim.x * 2];
	WORD* outputRow = (WORD*)&outMatrix[blockIdx.y * pitchOutMatrix + blockIdx.x * blockDim.x * 2];

	if (threadIdx.x == 0) {
		sharedMemory[0][threadIdx.x] = startOfProcessingRow[threadIdx.x];
		sharedMemory[1][threadIdx.x] = startOfProcessingRow[threadIdx.x + pitchInMatrix / 2];
		sharedMemory[2][threadIdx.x] = startOfProcessingRow[threadIdx.x + pitchInMatrix];

		/*if (blockIdx.y == 15) {
			printf("block: %d; thread: %d; line1: %d %d %d %d;\n", blockIdx.y, threadIdx.x, aa1.x, aa1.y, aa2.x, aa2.y);
		}*/
	}

	sharedMemory[0][threadIdx.x + 1] = startOfProcessingRow[threadIdx.x + 1];
	sharedMemory[1][threadIdx.x + 1] = startOfProcessingRow[threadIdx.x + 1 + pitchInMatrix / 2];
	sharedMemory[2][threadIdx.x + 1] = startOfProcessingRow[threadIdx.x + 1 + pitchInMatrix];

	__syncthreads();

	uchar2 aa1 = unpack(sharedMemory[0][threadIdx.x]);
	uchar2 aa2 = unpack(sharedMemory[0][threadIdx.x + 1]);

	uchar2 bb1 = unpack(sharedMemory[1][threadIdx.x]);
	uchar2 bb2 = unpack(sharedMemory[1][threadIdx.x + 1]);

	uchar2 cc1 = unpack(sharedMemory[2][threadIdx.x]);
	uchar2 cc2 = unpack(sharedMemory[2][threadIdx.x + 1]);

	uchar4 pixels;
	pixels.x = computeFirstPixel(aa1, aa2, bb1, bb2, cc1, cc2);
	pixels.y = computeSecondPixel(aa1, aa2, bb1, bb2, cc1, cc2);

	sharedMemoryOut[threadIdx.x] = pack(pixels);

	__syncthreads();

	outputRow[threadIdx.x] = sharedMemoryOut[threadIdx.x];
}

void cpu_filterImage(BYTE* primaryImage, BYTE* outputImage, int imageWidth, int imageHeight) {
	primaryImage = &primaryImage[imageWidth + 2 + 1];

	clock_t startTime, endTime;
	startTime = clock();
	for (auto i = 0; i < imageHeight; i++) {
		for (auto j = 0; j < imageWidth; j++) {
			short sum = 0;

			sum += primaryImage[i * (imageWidth + 2) + j];
			sum += primaryImage[i * (imageWidth + 2) + j + 1];
			sum += primaryImage[i * (imageWidth + 2) + j - 1];

			sum += primaryImage[(i + 1) * (imageWidth + 2) + j];
			sum += primaryImage[(i + 1) * (imageWidth + 2) + j + 1];
			sum += primaryImage[(i + 1) * (imageWidth + 2) + j - 1];

			sum += primaryImage[(i - 1) * (imageWidth + 2) + j];
			sum += primaryImage[(i - 1) * (imageWidth + 2) + j + 1];
			sum += primaryImage[(i - 1) * (imageWidth + 2) + j - 1];

			sum = sum / 9;

			if (sum > 255) {
				sum = 255;
			}
			else if (sum < 0) {
				sum = 0;
			}

			outputImage[i * imageWidth + j] = (unsigned char)sum;
		}
	}
	endTime = clock();
	printf("CPU time: %lf seconds\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);
}

void resizeImage(BYTE* primaryImage, BYTE* resizedImage, int imageWidth, int imageHeight) {
	for (int i = 0, int n = 0; i < imageHeight; i++, n++) {
		for (int j = 0, int m = 0; j < imageWidth; j++, m++) {
			resizedImage[n * (imageWidth + 2) + m] = primaryImage[i * imageWidth + j];

			if (j == 0 || j == imageWidth - 1) {
				m++;
				resizedImage[n * (imageWidth + 2) + m] = primaryImage[i * imageWidth + j];
			}
		}

		if (n == 0 || n == imageHeight) {
			i--;
		}
	}
}

void cuda_checkStatus(cudaError_t cudaStatus) {
	if (cudaStatus != cudaSuccess) {
		cout << "CUDA return error code: " << cudaStatus;
		cout << " " << cudaGetErrorString(cudaStatus) << endl;
		exit(-1);
	}
}

bool checkEquality(BYTE* firstImage, BYTE* secondImage, int imageWidth, int imageHeight) {
	for (int i = 0; i < imageWidth * imageHeight; i++) {
		if (fabs(firstImage[i] - secondImage[i]) > 1) {
			return false;
		}
	}
	return true;
}

void cuda_filterImage(BYTE* inMatrix, BYTE* outMatrix, bool optimizationFlag) {
	float resultTime;

	BYTE* device_inMatrix;
	BYTE* device_outMatrix;

	cudaEvent_t cuda_startTime;
	cudaEvent_t cuda_endTime;

	cuda_checkStatus(cudaEventCreate(&cuda_startTime));
	cuda_checkStatus(cudaEventCreate(&cuda_endTime));

	int numOfBlocksInRow = (int)ceil((double)(IMAGE_WIDTH) / (BLOCK_SIZE_X * 2));
	int numOfBlockInColumn = (int)ceil((double)(IMAGE_HEIGHT) / BLOCK_SIZE_Y);
	int blocksNeeded = numOfBlockInColumn * numOfBlocksInRow;
	int maxBlocksPerIteration = MAX_BLOCKS - MAX_BLOCKS % numOfBlocksInRow;

	for (int i = 0, int times = 1; i < blocksNeeded; i += maxBlocksPerIteration, times++) {
		int blocksInIteration = (blocksNeeded - i) < maxBlocksPerIteration ? blocksNeeded - i : maxBlocksPerIteration;
		size_t pitchInMatrix = 0, pitchOutMatrix = 0;
		int gridSizeY = blocksInIteration / numOfBlocksInRow;
		int gridSizeX = numOfBlocksInRow;

		cuda_checkStatus(cudaMallocPitch((void**)&device_inMatrix, &pitchInMatrix, IMAGE_WIDTH + 2, gridSizeY + 2));
		cuda_checkStatus(cudaMallocPitch((void**)&device_outMatrix, &pitchOutMatrix, IMAGE_WIDTH, gridSizeY));
		cuda_checkStatus(cudaMemcpy2D(
			device_inMatrix, pitchInMatrix,
			inMatrix, IMAGE_WIDTH + 2,
			IMAGE_WIDTH + 2, gridSizeY + 2,
			cudaMemcpyHostToDevice));

		dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
		dim3 gridSize(gridSizeX, gridSizeY);

		cuda_checkStatus(cudaEventRecord(cuda_startTime, NULL));

		if (optimizationFlag) {
			cuda_matrixSharedMemoryOperationKernel << < gridSize, blockSize >> > (device_inMatrix, device_outMatrix, pitchInMatrix, pitchOutMatrix);
		}
		else {
			cuda_matrixOperationKernel << < gridSize, blockSize >> > (device_inMatrix, device_outMatrix, pitchInMatrix, pitchOutMatrix);
		}

		cuda_checkStatus(cudaPeekAtLastError());
		cuda_checkStatus(cudaEventRecord(cuda_endTime, NULL));
		cuda_checkStatus(cudaEventSynchronize(cuda_endTime));

		cuda_checkStatus(cudaEventElapsedTime(&resultTime, cuda_startTime, cuda_endTime));

		if (optimizationFlag) {
			printf("%d: CUDA time with optimization: %lf seconds\n", times, (double)resultTime / CLOCKS_PER_SEC);
		}
		else {
			printf("%d: CUDA time: %lf seconds\n", times, (double)resultTime / CLOCKS_PER_SEC);
		}

		cuda_checkStatus(cudaMemcpy2D(
			outMatrix, IMAGE_WIDTH,
			device_outMatrix, pitchOutMatrix,
			IMAGE_WIDTH, gridSizeY,
			cudaMemcpyDeviceToHost)
		);

		inMatrix = &inMatrix[(IMAGE_WIDTH + 2) * gridSizeY * times];
		outMatrix = &outMatrix[IMAGE_WIDTH * gridSizeY * times];

		cuda_checkStatus(cudaFree(device_inMatrix));
		cuda_checkStatus(cudaFree(device_outMatrix));
	}
}