#include <cstdio>

__global__ void col_sum(const float* matrix, float* result,
                        int cols, int start_row, int end_row) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float sum = 0.0f;
        for (int r = start_row; r <= end_row; r++)
            sum += matrix[r * cols + col];
        result[col] = sum;
    }
}

void print_matrix(const float* mat, int rows, int cols) {
    printf("Matrix (%dx%d):\n", rows, cols);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++)
            printf("%6.1f ", mat[r * cols + c]);
        printf("\n");
    }
}

void print_array(const float* arr, int n, const char* label) {
    printf("%s: ", label);
    for (int i = 0; i < n; i++)
        printf("%6.1f ", arr[i]);
    printf("\n");
}

int main() {
    int rows = 3, cols = 5;

    float matrix[3][5] = {
        {2, 3, 3, 4, 4},
        {1, 6, 7, 4, 4},
        {5, 2, 1, 3, 8}
    };

    int start_row = 0, end_row = 2;

    print_matrix((float*)matrix, rows, cols);
    printf("\nSumming rows %d to %d (inclusive) column-wise:\n\n", start_row, end_row);

    float* d_matrix;
    float* d_result;
    float h_result[5];

    int mat_size = rows * cols * sizeof(float);
    int res_size = cols * sizeof(float);

    cudaMalloc(&d_matrix, mat_size);
    cudaMalloc(&d_result, res_size);
    cudaMemcpy(d_matrix, matrix, mat_size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (cols + threads - 1) / threads;

    col_sum<<<blocks, threads>>>(d_matrix, d_result, cols, start_row, end_row);
    cudaDeviceSynchronize();

    cudaMemcpy(h_result, d_result, res_size, cudaMemcpyDeviceToHost);

    print_array(h_result, cols, "Column sums");

    cudaFree(d_matrix);
    cudaFree(d_result);

    return 0;
}
