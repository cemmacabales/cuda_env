#include <iostream>
#include <cmath>

// host - cpu
// device - gpu

__global__ void vector_add(const float* A, float* B, float* C, int N){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (i < N) {
		C[i] = A[i] + B[i];
	}	
}

int main(){
	const int N = 10;
	float A[N], B[N], C[N];
	
	for (int i = 0; i < N; ++i){
		A[i] = (float)i + 1.0f;
		B[i] = 2.0f;
	}
	
	float *d_a, *d_b, *d_c;
	cudaMalloc(&d_a, N * sizeof(float));
	cudaMalloc(&d_b, N * sizeof(float));
	cudaMalloc(&d_c, N * sizeof(float));
	
	cudaMemcpy(d_a, A, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, B, N * sizeof(float), cudaMemcpyHostToDevice);
	
	int blocksize = 256;
	int gridsize = (N + blocksize - 1) / blocksize;
	
	vector_add<<<gridsize, blocksize>>>(d_a, d_b, d_c, N);
	
	cudaDeviceSynchronize();
	cudaMemcpy(C, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
	
	printf("Vector A: ");
	for(int i = 0; i < N; ++i){
		std::cout << A[i] << ", ";
	}
	std::cout << "\n";
	
	printf("Vector B: ");
	for(int i = 0; i < N; ++i){
		std::cout << B[i] << ", ";
	}
	std::cout << "\n";
	
	printf("Vector C (Result): ");
	for(int i = 0; i < N; ++i){
		std::cout << C[i] << ", ";
	}
	std::cout << "\n";

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);	
	
	return 0;
}
