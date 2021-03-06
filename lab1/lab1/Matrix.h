#pragma once
#include <stdlib.h>
#include <time.h>
#include <emmintrin.h>
#include <iostream>
using namespace std;

template <typename T>
class Matrix
{
private:
	int mainMatrixWidth;
	int mainMatrixHeight;
	int cellMatrixWidth;
	int cellMatrixHeight;
	T**** matrixPointer;
public:
	Matrix(int mainMatrixWidth, int mainMatrixHeight, int cellMatrixWidth, int cellMatrixHeight)
	{
		this->mainMatrixWidth = mainMatrixWidth;
		this->mainMatrixHeight = mainMatrixHeight;
		this->cellMatrixWidth = cellMatrixWidth;
		this->cellMatrixHeight = cellMatrixHeight;

		this->matrixPointer = nullptr;

		try {
			matrixPointer = new T***[mainMatrixHeight];

			for (int i = 0; i < mainMatrixHeight; i++) {
				matrixPointer[i] = new T**[mainMatrixWidth];
				for (int j = 0; j < mainMatrixWidth; j++) {
					matrixPointer[i][j] = new T*[cellMatrixHeight];
					for (int k = 0; k < cellMatrixHeight; k++) {
						matrixPointer[i][j][k] = new T[cellMatrixWidth];
						for (int m = 0; m < cellMatrixWidth; m++) {
							matrixPointer[i][j][k][m] = 0;
						}
					}
				}
			}
		}
		catch (...) {
			matrixPointer = nullptr;
		}
	}

	Matrix() {}

	bool equals(Matrix<T> &matrix) {
		for (int i = 0; i < mainMatrixHeight; i++) {
			for (int j = 0; j < mainMatrixWidth; j++) {
				for (int k = 0; k < cellMatrixHeight; k++) {
					for (int l = 0; l < cellMatrixWidth; l++) {
						if (this->matrixPointer[i][j][k][l] != matrix.getMatrixPointer()[i][j][k][l]) {
							return false;
						}
					}
				}
			}
		}
		return true;
	}

	Matrix<T> multiplyVectorized(Matrix<T> &matrixB) {
		if (this->mainMatrixWidth != matrixB.getMainMatrixHeight()
			|| this->cellMatrixWidth != matrixB.getCellMatrixHeight())
		{
			return matrixB;
		}

		Matrix<T> resultMatrix(this->mainMatrixHeight, matrixB.getMainMatrixWidth(), this->cellMatrixHeight, matrixB.getCellMatrixWidth());
		
		for (int m = 0; m < this->mainMatrixHeight; m++) {
			for (int n = 0; n < this->mainMatrixWidth; n++) {
				for (int y = 0; y < matrixB.getMainMatrixWidth(); y++) {
					mulMatrixVectorized(matrixB, resultMatrix, m, n, y);
				}
			}
		}

		return resultMatrix;
	}

	void mulMatrixVectorized(Matrix<T> &matrixB, Matrix<T> &resultMatrix, int m, int n, int y) // �������� � ��������� �����, ����� ���������������, ����� ������ 1204 (���� 0 ��� ��� ������)
	{
		T* __restrict resultMatrixCellRow = nullptr;
		T* __restrict matrixBCellRow = nullptr;

		for (int i = 0; i < this->cellMatrixHeight; i++) {
			resultMatrixCellRow = resultMatrix.getMatrixPointer()[m][n][i];
			for (int j = 0; j < this->cellMatrixWidth; j++) {
				matrixBCellRow = matrixB.getMatrixPointer()[m][y][j];
#pragma loop(hint_parallel(0))
				for (int k = 0; k < matrixB.getCellMatrixWidth(); k++) {
					resultMatrixCellRow[k] += this->matrixPointer[m][n][i][j] * matrixBCellRow[k];
				}
			}
		}
	}

	Matrix<T> multiplyNotVectorized(Matrix<T> &matrixB) {
		if (this->mainMatrixWidth != matrixB.getMainMatrixHeight()
			|| this->cellMatrixWidth != matrixB.getCellMatrixHeight())
		{
			return matrixB;
		}

		Matrix<T> resultMatrix(this->mainMatrixHeight, matrixB.getMainMatrixWidth(), this->cellMatrixHeight, matrixB.getCellMatrixWidth());

		for (int m = 0; m < this->mainMatrixHeight; m++) {
			for (int n = 0; n < this->mainMatrixWidth; n++) {
				for (int y = 0; y < matrixB.getMainMatrixWidth(); y++) {
					mulMatrixNotVectorized(matrixB, resultMatrix, m, n, y);
				}
			}
		}

		return resultMatrix;
	}

	void mulMatrixNotVectorized(Matrix<T> &matrixB, Matrix<T> &resultMatrix, int m, int n, int y)
	{
		T* __restrict resultMatrixCellRow = nullptr;
		T* __restrict matrixBCellRow = nullptr;

		for (int i = 0; i < this->cellMatrixHeight; i++) {
			resultMatrixCellRow = resultMatrix.getMatrixPointer()[m][n][i];
			for (int j = 0; j < this->cellMatrixWidth; j++) {
				matrixBCellRow = matrixB.getMatrixPointer()[m][y][j];
#pragma loop(no_vector)
				for (int k = 0; k < matrixB.getCellMatrixWidth(); k++) {
					resultMatrixCellRow[k] += this->matrixPointer[m][n][i][j] * matrixBCellRow[k];
				}
			}
		}
	}

	Matrix<T> multiplySse(Matrix<T> &matrixB) {
		if (this->mainMatrixWidth != matrixB.getMainMatrixHeight()
			|| this->cellMatrixWidth != matrixB.getCellMatrixHeight())
		{
			return matrixB;
		}

		Matrix<T> resultMatrix(this->mainMatrixHeight, matrixB.getMainMatrixWidth(), this->cellMatrixHeight, matrixB.getCellMatrixWidth());

		for (int m = 0; m < this->mainMatrixHeight; m++) {
			for (int n = 0; n < this->mainMatrixWidth; n++) {
				for (int y = 0; y < matrixB.getMainMatrixWidth(); y++) {
					mulSse(matrixB, resultMatrix, m, n, y);
				}
			}
		}

		return resultMatrix;
	}

	void mulSse(Matrix<T> &matrixB, Matrix<T> &resultMatrix, int m, int n, int y)
	{
		T* __restrict resultMatrixCellRow = nullptr;
		T* __restrict matrixBCellRow = nullptr;

		__m128 resultMatrixReg;
		__m128 valueFromMatrixA;
		__m128 matrixBRowReg;

		for (int i = 0; i < this->cellMatrixHeight; i++)
		{
			resultMatrixCellRow = resultMatrix.getMatrixPointer()[m][n][i];
			for (int j = 0; j < this->cellMatrixWidth; j++)
			{
				matrixBCellRow = matrixB.getMatrixPointer()[m][y][j];
				valueFromMatrixA = _mm_set1_ps(this->matrixPointer[m][n][i][j]);
				for (int k = 0; k < matrixB.getCellMatrixWidth(); k += 4)
				{
					resultMatrixReg = _mm_load_ps(resultMatrixCellRow + k);
					matrixBRowReg = _mm_load_ps(matrixBCellRow + k);
					resultMatrixReg = _mm_add_ps(resultMatrixReg, _mm_mul_ps(valueFromMatrixA, matrixBRowReg));
					_mm_store_ps(resultMatrixCellRow + k, resultMatrixReg);
				}
			}
		}
	}

	bool generateValues() {
		srand((unsigned)time(NULL));

		try {
			for (int i = 0; i < this->mainMatrixHeight; i++) {
				for (int j = 0; j < mainMatrixWidth; j++) {
					for (int k = 0; k < cellMatrixHeight; k++) {
						for (int m = 0; m < cellMatrixWidth; m++) {
							matrixPointer[i][j][k][m] = (T)(rand() % 1000);
						}
					}
				}
			}
		}
		catch (...) {
			return false;
		}

		return true;
	}

	T**** getMatrixPointer()
	{
		return this->matrixPointer;
	}

	int getMainMatrixWidth()
	{
		return this->mainMatrixWidth;
	}

	int getMainMatrixHeight()
	{
		return this->mainMatrixHeight;
	}

	int getCellMatrixWidth()
	{
		return this->cellMatrixWidth;
	}

	int getCellMatrixHeight()
	{
		return this->cellMatrixHeight;
	}
};