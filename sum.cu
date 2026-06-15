#include <cstdio>
__global__ void sum(float *d_A, int n) {
    int t = n/2;
    int s = n%2 ? t+1 : t;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;  
    if(tid < t) {
        d_A[tid] = d_A[tid] + d_A[tid+s];
    }
}

void print(float *A, int n){
    printf("A: \n");
    for (int x = 0; x < n; x++) {
        printf("%6.1f ", A[x]);
    }
    printf("\n");
}




void total(int N, float *A){
    int n =N;
    dim3 threadsperblock(2);
    dim3 nummblocks(3);

    printf("Starting");

    int size = N * sizeof(float);
    float *d_A;
    
    cudaMalloc((void**)&d_A, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    while(n>1){        
        printf("n = %d\n", n);
        print(A, n);
        sum<<<threadsperblock, nummblocks>>>(d_A, n);

        n = (n % 2 == 0) ? n/2 : n/2 +1;

        cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost);

    }
}

int main() {
    float A[] = {2,3,3,4,4,1,6,7,4,4}; 
    total(10, A); 
    printf("Total = %6.1f ", A[0]);
    return 0;
}

