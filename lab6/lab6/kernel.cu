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
#include <tuple>

#pragma comment(lib, "cudart") 

#define BLOCK_SIZE_X 1024
#define BLOCK_SIZE_Y 1
//#define IMAGE_WIDTH 11880
//#define IMAGE_HEIGHT 8648
#define IMAGE_WIDTH 6000
#define IMAGE_HEIGHT 4000
#define MAX_BLOCKS 50000

using namespace std;

void cpu_filterImage(BYTE*, BYTE*, int, int);
void cuda_filterImage(BYTE*, BYTE*, bool);
void cuda_checkStatus(cudaError_t);
void resizeImage(BYTE*, BYTE*, int, int);
bool checkEquality(BYTE*, BYTE*, int, int);

int main() {
	unsigned int imageWidth = 0, imageHeight = 0, channels;

	const char primaryImagePath[] = "D:\\big.ppm";
	const char outputImageCpuPath[] = "D:\\imageCPU.ppm";
	const char outputImageGpuPath[] = "D:\\imageGPU.ppm";
	const char outputImageGpuSharedPath[] = "D:\\imageGPUshared.ppm";

	BYTE *primaryImage = NULL;

	__loadPPM(primaryImagePath, &primaryImage, &imageWidth, &imageHeight, &channels);

	auto* outputImageCpu = (BYTE*)malloc(imageWidth * imageHeight * sizeof(BYTE) * 3);
	auto* outputImageGpu = (BYTE*)malloc(imageWidth * imageHeight * sizeof(BYTE) * 3);
	auto* outputImageGpuShared = (BYTE*)malloc(imageWidth * imageHeight * sizeof(BYTE) * 3);
	auto* resizedImage = (BYTE*)malloc((imageWidth + 2) * (imageHeight + 2) * sizeof(BYTE) * 3);

	resizeImage(primaryImage, resizedImage, imageWidth, imageHeight);

	/*for (size_t i = 0; i < (imageWidth + 2) * (imageHeight + 2); i++)
	{
		printf("%-3d:%-3d:%-3d ", resizedImage[i * 3], resizedImage[i * 3 + 1], resizedImage[i * 3 + 2]);
		if ((i + 1) % (imageWidth + 2) == 0) {
			cout << endl;
		}
	}*/

	cpu_filterImage(resizedImage, outputImageCpu, imageWidth, imageHeight);
	cuda_filterImage(resizedImage, outputImageGpu, false);
	cuda_filterImage(resizedImage, outputImageGpuShared, true);

	__savePPM(outputImageCpuPath, outputImageCpu, imageWidth, imageHeight, channels);
	__savePPM(outputImageGpuPath, outputImageGpu, imageWidth, imageHeight, channels);
	__savePPM(outputImageGpuSharedPath, outputImageGpuShared, imageWidth, imageHeight, channels);

	cout << "Start compare" << endl;

	if (checkEquality(outputImageCpu, outputImageGpu, IMAGE_WIDTH, IMAGE_HEIGHT) &&
		checkEquality(outputImageGpu, outputImageGpuShared, IMAGE_WIDTH, IMAGE_HEIGHT)) {
		cout << "Results are equals!" << endl;
	}
	else {
		cout << "Results are NOT equals!" << endl;
	}

	free(primaryImage);
	free(outputImageGpu);
	free(outputImageCpu);
	free(resizedImage);
}

__device__ WORD pack(uchar3 pixelLine)
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

/*__device__ tuple<uchar3, uchar3> computeNewPixels(
	uchar2 aa1, uchar2 aa2, uchar2 aa3, uchar2 aa4, uchar2 aa5, uchar2 aa6,
	uchar2 bb1, uchar2 bb2, uchar2 bb3, uchar2 bb4, uchar2 bb5, uchar2 bb6,
	uchar2 cc1, uchar2 cc2, uchar2 cc3, uchar2 cc4, uchar2 cc5, uchar2 cc6) {

	uchar3 rgb1, rgb2;

	rgb1.x = sumColors(aa1.x, aa2.y, aa4.x, bb1.x, bb2.y, bb4.x, cc1.x, cc2.y, cc4.x);
	rgb1.y = sumColors(aa1.y, aa3.x, aa4.y, bb1.y, bb3.x, bb4.y, cc1.y, cc3.x, cc4.y);
	rgb1.z = sumColors(aa2.x, aa3.y, aa5.x, bb2.x, bb3.y, bb5.x, cc2.x, cc3.y, cc5.x);

	rgb2.x = sumColors(aa2.y, aa4.x, aa5.y, bb2.y, bb4.x, bb5.y, cc2.y, cc4.x, cc5.y);
	rgb2.y = sumColors(aa3.x, aa4.y, aa6.x, bb3.x, bb4.y, bb6.x, cc3.x, cc4.y, cc6.x);
	rgb2.z = sumColors(aa3.y, aa5.x, aa6.y, bb3.y, bb5.x, bb6.y, cc3.y, cc5.x, cc6.y);

	return make_tuple(rgb1, rgb2);
}*/

__device__ BYTE sumColors(BYTE a1, BYTE a2, BYTE a3, BYTE a4, BYTE a5, BYTE a6, BYTE a7, BYTE a8, BYTE a9)
{
	uint32_t result = 0;

	result = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9;

	result /= 9;

	if (result > 255) {
		result = 255;
	}

	return (BYTE)result;
}

__device__ uchar3 computeFirstPixel(
	uchar2 aa1, uchar2 aa2, uchar2 aa3, uchar2 aa4, uchar2 aa5, uchar2 aa6,
	uchar2 bb1, uchar2 bb2, uchar2 bb3, uchar2 bb4, uchar2 bb5, uchar2 bb6,
	uchar2 cc1, uchar2 cc2, uchar2 cc3, uchar2 cc4, uchar2 cc5, uchar2 cc6)
{
	uint32_t result = 0;
	uchar3 rgb;

	result += aa1.x;
	result += aa2.y;
	result += aa4.x;

	result += bb1.x;
	result += bb2.y;
	result += bb4.x;

	result += cc1.x;
	result += cc2.y;
	result += cc4.x;

	result /= 9;

	if (result > 255) {
		result = 255;
	}

	rgb.x = result;

	result = 0;

	result += aa1.y;
	result += aa3.x;
	result += aa4.y;

	result += bb1.y;
	result += bb3.x;
	result += bb4.y;

	result += cc1.y;
	result += cc3.x;
	result += cc4.y;

	result /= 9;

	if (result > 255) {
		result = 255;
	}

	rgb.y = result;

	result = 0;

	result += aa2.x;
	result += aa3.y;
	result += aa5.x;

	result += bb2.x;
	result += bb3.y;
	result += bb5.x;

	result += cc2.x;
	result += cc3.y;
	result += cc5.x;

	result /= 9;

	if (result > 255) {
		result = 255;
	}

	rgb.z = result;

	return rgb;
}

__device__ uchar3 computeSecondPixel(
	uchar2 aa1, uchar2 aa2, uchar2 aa3, uchar2 aa4, uchar2 aa5, uchar2 aa6,
	uchar2 bb1, uchar2 bb2, uchar2 bb3, uchar2 bb4, uchar2 bb5, uchar2 bb6,
	uchar2 cc1, uchar2 cc2, uchar2 cc3, uchar2 cc4, uchar2 cc5, uchar2 cc6)
{
	uint32_t result = 0;
	uchar3 rgb;

	result += aa2.y;
	result += aa4.x;
	result += aa5.y;

	result += bb2.y;
	result += bb4.x;
	result += bb5.y;

	result += cc2.y;
	result += cc4.x;
	result += cc5.y;

	result /= 9;

	if (result > 255) {
		result = 255;
	}

	rgb.x = result;

	result = 0;

	result += aa3.x;
	result += aa4.y;
	result += aa6.x;

	result += bb3.x;
	result += bb4.y;
	result += bb6.x;

	result += cc3.x;
	result += cc4.y;
	result += cc6.x;

	result /= 9;

	if (result > 255) {
		result = 255;
	}

	rgb.y = result;

	result = 0;

	result += aa3.y;
	result += aa5.x;
	result += aa6.y;

	result += bb3.y;
	result += bb5.x;
	result += bb6.y;

	result += cc3.y;
	result += cc5.x;
	result += cc6.y;

	result /= 9;

	if (result > 255) {
		result = 255;
	}

	rgb.z = result;

	return rgb;
}

__global__ void cuda_matrixOperationKernel(BYTE* inMatrix, BYTE* outMatrix, size_t pitchInMatrix, size_t pitchOutMatrix) {
	int remainderElements = (IMAGE_WIDTH % (blockDim.x * 2)) / 2;

	if (remainderElements != 0 && (blockIdx.x + 1) % gridDim.x == 0 && threadIdx.x >= remainderElements) {
		return;
	}

	WORD *startOfProcessingRow = (WORD*)&inMatrix[pitchInMatrix * blockIdx.y + blockIdx.x * blockDim.x * 2 * 3 + threadIdx.x * 2 * 3];

	WORD a1 = startOfProcessingRow[0];
	WORD a2 = startOfProcessingRow[1];
	WORD a3 = startOfProcessingRow[2];
	WORD a4 = startOfProcessingRow[3];
	WORD a5 = startOfProcessingRow[4];
	WORD a6 = startOfProcessingRow[5];

	WORD b1 = startOfProcessingRow[pitchInMatrix / 2];
	WORD b2 = startOfProcessingRow[pitchInMatrix / 2 + 1];
	WORD b3 = startOfProcessingRow[pitchInMatrix / 2 + 2];
	WORD b4 = startOfProcessingRow[pitchInMatrix / 2 + 3];
	WORD b5 = startOfProcessingRow[pitchInMatrix / 2 + 4];
	WORD b6 = startOfProcessingRow[pitchInMatrix / 2 + 5];

	WORD c1 = startOfProcessingRow[pitchInMatrix];
	WORD c2 = startOfProcessingRow[pitchInMatrix + 1];
	WORD c3 = startOfProcessingRow[pitchInMatrix + 2];
	WORD c4 = startOfProcessingRow[pitchInMatrix + 3];
	WORD c5 = startOfProcessingRow[pitchInMatrix + 4];
	WORD c6 = startOfProcessingRow[pitchInMatrix + 5];

	uchar2 aa1 = unpack(a1);
	uchar2 aa2 = unpack(a2);
	uchar2 aa3 = unpack(a3);
	uchar2 aa4 = unpack(a4);
	uchar2 aa5 = unpack(a5);
	uchar2 aa6 = unpack(a6);

	uchar2 bb1 = unpack(b1);
	uchar2 bb2 = unpack(b2);
	uchar2 bb3 = unpack(b3);
	uchar2 bb4 = unpack(b4);
	uchar2 bb5 = unpack(b5);
	uchar2 bb6 = unpack(b6);

	uchar2 cc1 = unpack(c1);
	uchar2 cc2 = unpack(c2);
	uchar2 cc3 = unpack(c3);
	uchar2 cc4 = unpack(c4);
	uchar2 cc5 = unpack(c5);
	uchar2 cc6 = unpack(c6);

	uchar3 firstPixel = computeFirstPixel(aa1, aa2, aa3, aa4, aa5, aa6, bb1, bb2, bb3, bb4, bb5, bb6, cc1, cc2, cc3, cc4, cc5, cc6);
	uchar3 secondPixel = computeSecondPixel(aa1, aa2, aa3, aa4, aa5, aa6, bb1, bb2, bb3, bb4, bb5, bb6, cc1, cc2, cc3, cc4, cc5, cc6);
	//uchar3 firstPixel, secondPixel;
	//tie(firstPixel, secondPixel) = computeNewPixels(aa1, aa2, aa3, aa4, aa5, aa6, bb1, bb2, bb3, bb4, bb5, bb6, cc1, cc2, cc3, cc4, cc5, cc6);
	outMatrix = &outMatrix[blockIdx.y * pitchOutMatrix + threadIdx.x * 2 * 3 + blockIdx.x * blockDim.x * 2 * 3];
	outMatrix[0] = firstPixel.x;
	outMatrix[1] = firstPixel.y;
	outMatrix[2] = firstPixel.z;
	outMatrix[3] = secondPixel.x;
	outMatrix[4] = secondPixel.y;
	outMatrix[5] = secondPixel.z;
}

__global__ void cuda_matrixSharedMemoryOperationKernel(BYTE* inMatrix, BYTE* outMatrix, size_t pitchInMatrix, size_t pitchOutMatrix) {
	int remainderElements = (IMAGE_WIDTH % (blockDim.x * 2)) / 2;

	if (remainderElements != 0 && (blockIdx.x + 1) % gridDim.x == 0 && threadIdx.x >= remainderElements) {
		return;
	}

	__shared__ WORD sharedMemory[3][(BLOCK_SIZE_X + 1) * 3];
	__shared__ WORD sharedMemoryOut[BLOCK_SIZE_X * 3];

	WORD* startOfProcessingRow = (WORD*)&inMatrix[blockIdx.y * pitchInMatrix + blockIdx.x * blockDim.x * 2 * 3];
	WORD* outputRow = (WORD*)&outMatrix[blockIdx.y * pitchOutMatrix + blockIdx.x * blockDim.x * 2 * 3];

	if (threadIdx.x == 0) {
		sharedMemory[0][threadIdx.x] = startOfProcessingRow[threadIdx.x];
		sharedMemory[0][threadIdx.x + 1] = startOfProcessingRow[threadIdx.x + 1];
		sharedMemory[0][threadIdx.x + 2] = startOfProcessingRow[threadIdx.x + 2];

		sharedMemory[1][threadIdx.x] = startOfProcessingRow[threadIdx.x + pitchInMatrix / 2];
		sharedMemory[1][threadIdx.x + 1] = startOfProcessingRow[threadIdx.x + pitchInMatrix / 2 + 1];
		sharedMemory[1][threadIdx.x + 2] = startOfProcessingRow[threadIdx.x + pitchInMatrix / 2 + 2];

		sharedMemory[2][threadIdx.x] = startOfProcessingRow[threadIdx.x + pitchInMatrix];
		sharedMemory[2][threadIdx.x + 1] = startOfProcessingRow[threadIdx.x + pitchInMatrix + 1];
		sharedMemory[2][threadIdx.x + 2] = startOfProcessingRow[threadIdx.x + pitchInMatrix + 2];


		/*if (blockIdx.y == 15) {
			printf("block: %d; thread: %d; line1: %d %d %d %d;\n", blockIdx.y, threadIdx.x, aa1.x, aa1.y, aa2.x, aa2.y);
		}*/
	}

	sharedMemory[0][threadIdx.x * 3 + 3] = startOfProcessingRow[threadIdx.x * 3 + 3];
	sharedMemory[0][threadIdx.x * 3 + 4] = startOfProcessingRow[threadIdx.x * 3 + 4];
	sharedMemory[0][threadIdx.x * 3 + 5] = startOfProcessingRow[threadIdx.x * 3 + 5];

	sharedMemory[1][threadIdx.x * 3 + 3] = startOfProcessingRow[threadIdx.x * 3 + 3 + pitchInMatrix / 2];
	sharedMemory[1][threadIdx.x * 3 + 4] = startOfProcessingRow[threadIdx.x * 3 + 4 + pitchInMatrix / 2];
	sharedMemory[1][threadIdx.x * 3 + 5] = startOfProcessingRow[threadIdx.x * 3 + 5 + pitchInMatrix / 2];

	sharedMemory[2][threadIdx.x * 3 + 3] = startOfProcessingRow[threadIdx.x * 3 + 3 + pitchInMatrix];
	sharedMemory[2][threadIdx.x * 3 + 4] = startOfProcessingRow[threadIdx.x * 3 + 4 + pitchInMatrix];
	sharedMemory[2][threadIdx.x * 3 + 5] = startOfProcessingRow[threadIdx.x * 3 + 5 + pitchInMatrix];


	__syncthreads();

	WORD a1 = sharedMemory[0][threadIdx.x * 3];
	WORD a2 = sharedMemory[0][threadIdx.x * 3 + 1];
	WORD a3 = sharedMemory[0][threadIdx.x * 3 + 2];
	WORD a4 = sharedMemory[0][threadIdx.x * 3 + 3];
	WORD a5 = sharedMemory[0][threadIdx.x * 3 + 4];
	WORD a6 = sharedMemory[0][threadIdx.x * 3 + 5];

	WORD b1 = sharedMemory[1][threadIdx.x * 3];
	WORD b2 = sharedMemory[1][threadIdx.x * 3 + 1];
	WORD b3 = sharedMemory[1][threadIdx.x * 3 + 2];
	WORD b4 = sharedMemory[1][threadIdx.x * 3 + 3];
	WORD b5 = sharedMemory[1][threadIdx.x * 3 + 4];
	WORD b6 = sharedMemory[1][threadIdx.x * 3 + 5];

	WORD c1 = sharedMemory[2][threadIdx.x * 3];
	WORD c2 = sharedMemory[2][threadIdx.x * 3 + 1];
	WORD c3 = sharedMemory[2][threadIdx.x * 3 + 2];
	WORD c4 = sharedMemory[2][threadIdx.x * 3 + 3];
	WORD c5 = sharedMemory[2][threadIdx.x * 3 + 4];
	WORD c6 = sharedMemory[2][threadIdx.x * 3 + 5];

	uchar2 aa1 = unpack(a1);
	uchar2 aa2 = unpack(a2);
	uchar2 aa3 = unpack(a3);
	uchar2 aa4 = unpack(a4);
	uchar2 aa5 = unpack(a5);
	uchar2 aa6 = unpack(a6);

	uchar2 bb1 = unpack(b1);
	uchar2 bb2 = unpack(b2);
	uchar2 bb3 = unpack(b3);
	uchar2 bb4 = unpack(b4);
	uchar2 bb5 = unpack(b5);
	uchar2 bb6 = unpack(b6);

	uchar2 cc1 = unpack(c1);
	uchar2 cc2 = unpack(c2);
	uchar2 cc3 = unpack(c3);
	uchar2 cc4 = unpack(c4);
	uchar2 cc5 = unpack(c5);
	uchar2 cc6 = unpack(c6);

	uchar3 firstPixel = computeFirstPixel(aa1, aa2, aa3, aa4, aa5, aa6, bb1, bb2, bb3, bb4, bb5, bb6, cc1, cc2, cc3, cc4, cc5, cc6);
	uchar3 secondPixel = computeSecondPixel(aa1, aa2, aa3, aa4, aa5, aa6, bb1, bb2, bb3, bb4, bb5, bb6, cc1, cc2, cc3, cc4, cc5, cc6);

	sharedMemoryOut[threadIdx.x * 3] = ((firstPixel.y << 8) | firstPixel.x);
	sharedMemoryOut[threadIdx.x * 3 + 1] = ((secondPixel.x << 8) | firstPixel.z);
	sharedMemoryOut[threadIdx.x * 3 + 2] = ((secondPixel.z << 8) | secondPixel.y);

	__syncthreads();

	outputRow[threadIdx.x * 3] = sharedMemoryOut[threadIdx.x * 3];
	outputRow[threadIdx.x * 3 + 1] = sharedMemoryOut[threadIdx.x * 3 + 1];
	outputRow[threadIdx.x * 3 + 2] = sharedMemoryOut[threadIdx.x * 3 + 2];

}

void cpu_filterImage(BYTE* primaryImage, BYTE* outputImage, int imageWidth, int imageHeight) {
	primaryImage = &primaryImage[(imageWidth + 2 + 1) * 3];

	clock_t startTime, endTime;
	startTime = clock();
	for (auto i = 0; i < imageHeight; i++) {
		for (auto j = 0; j < imageWidth; j++) {
			for (auto k = 0; k < 3; k++) {
				short sum = 0;
				int index = 0;

				index = (i * (imageWidth + 2) + j) * 3 + k;
				sum += primaryImage[(i * (imageWidth + 2) + j) * 3 + k];
				sum += primaryImage[(i * (imageWidth + 2) + j + 1) * 3 + k];
				sum += primaryImage[(i * (imageWidth + 2) + j - 1) * 3 + k];

				sum += primaryImage[((i + 1) * (imageWidth + 2) + j) * 3 + k];
				sum += primaryImage[((i + 1) * (imageWidth + 2) + j + 1) * 3 + k];
				sum += primaryImage[((i + 1) * (imageWidth + 2) + j - 1) * 3 + k];

				sum += primaryImage[((i - 1) * (imageWidth + 2) + j) * 3 + k];
				sum += primaryImage[((i - 1) * (imageWidth + 2) + j + 1) * 3 + k];
				sum += primaryImage[((i - 1) * (imageWidth + 2) + j - 1) * 3 + k];

				sum = sum / 9;

				if (sum > 255) {
					sum = 255;
				}

				outputImage[(i * imageWidth + j) * 3 + k] = (BYTE)sum;
			}
		}
	}
	endTime = clock();
	printf("CPU time: %lf seconds\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);
}

void resizeImage(BYTE* primaryImage, BYTE* resizedImage, int imageWidth, int imageHeight) {
	for (int i = 0, int n = 0; i < imageHeight; i++, n++) {
		for (int j = 0, int m = 0; j < imageWidth; j++, m++) {
			resizedImage[(n * (imageWidth + 2) + m) * 3] = primaryImage[(i * imageWidth + j) * 3];
			resizedImage[(n * (imageWidth + 2) + m) * 3 + 1] = primaryImage[(i * imageWidth + j) * 3 + 1];
			resizedImage[(n * (imageWidth + 2) + m) * 3 + 2] = primaryImage[(i * imageWidth + j) * 3 + 2];

			if (j == 0 || j == imageWidth - 1) {
				m++;
				resizedImage[(n * (imageWidth + 2) + m) * 3] = primaryImage[(i * imageWidth + j) * 3];
				resizedImage[(n * (imageWidth + 2) + m) * 3 + 1] = primaryImage[(i * imageWidth + j) * 3 + 1];
				resizedImage[(n * (imageWidth + 2) + m) * 3 + 2] = primaryImage[(i * imageWidth + j) * 3 + 2];
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

		cuda_checkStatus(cudaMallocPitch((void**)&device_inMatrix, &pitchInMatrix, (IMAGE_WIDTH + 2) * 3, gridSizeY + 2));
		cuda_checkStatus(cudaMallocPitch((void**)&device_outMatrix, &pitchOutMatrix, IMAGE_WIDTH * 3, gridSizeY));
		cuda_checkStatus(cudaMemcpy2D(
			device_inMatrix, pitchInMatrix,
			inMatrix, (IMAGE_WIDTH + 2) * 3,
			(IMAGE_WIDTH + 2) * 3, gridSizeY + 2,
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
			outMatrix, IMAGE_WIDTH * 3,
			device_outMatrix, pitchOutMatrix,
			IMAGE_WIDTH * 3, gridSizeY,
			cudaMemcpyDeviceToHost)
		);

		inMatrix = &inMatrix[(IMAGE_WIDTH + 2) * gridSizeY * times * 3];
		outMatrix = &outMatrix[IMAGE_WIDTH * gridSizeY * times * 3];

		cuda_checkStatus(cudaFree(device_inMatrix));
		cuda_checkStatus(cudaFree(device_outMatrix));
	}
}