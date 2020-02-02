#include <Windows.h>
#include <iostream>
#include "Matrix.h"
using namespace std;

typedef float TYPE;

#define MAIN_WIDTH 2000
#define MAIN_HEIGHT 2000
#define MATRIX_A_CELL_WIDTH 8
#define MATRIX_A_CELL_HEIGHT 4
#define MATRIX_B_CELL_WIDTH 4
#define MATRIX_B_CELL_HEIGHT 8

int main() {
	Matrix<TYPE> matrixA(MAIN_WIDTH, MAIN_HEIGHT, MATRIX_A_CELL_WIDTH, MATRIX_A_CELL_HEIGHT),
		matrixB(MAIN_WIDTH, MAIN_HEIGHT, MATRIX_B_CELL_WIDTH, MATRIX_B_CELL_HEIGHT),
		matrixC, matrixD, matrixF;

	matrixA.generateValues();
	matrixB.generateValues();

	ULONGLONG resultTimeSse;
	ULONGLONG resultTime;
	ULONGLONG resultTimeNotVectorized;

	ULONGLONG endTime;

	ULONGLONG startTime = GetTickCount64();
	matrixC = matrixA.multiplyVectorized(matrixB);
	endTime = GetTickCount64();

	resultTime = endTime - startTime;

	startTime = GetTickCount64();
	matrixD = matrixA.multiplyNotVectorized(matrixB);
	endTime = GetTickCount64();

	resultTimeNotVectorized = endTime - startTime;

	startTime = GetTickCount64();
	matrixD = matrixA.multiplySse(matrixB);
	endTime = GetTickCount64();

	resultTimeSse = endTime - startTime;

	cout << "Vectorization Ticks: " << resultTime << "." << endl;
	cout << "SSE Ticks: " << resultTimeSse << "." << endl;
	cout << "Not Vectorized Ticks: " << resultTimeNotVectorized << "." << endl << endl;

	cout << "Vectorized speed up: x" << (double)resultTimeNotVectorized / (double)resultTime << "." << endl;
	cout << "SSE speed up: x" << (double)resultTimeNotVectorized / (double)resultTimeSse << ".\n" << endl;

	if (matrixC.equals(matrixD)) {
		cout << "Matrices by SSE and by Vectorization are equal." << endl;
	}
	else {
		cout << "Matrices by SSE and by Vectorization are not equal." << endl;
	}

	system("pause");
	return 0;
}