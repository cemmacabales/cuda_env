#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define ROWS 3
#define COLS 4
#define THREADS 256

static bool USE_CUDA = false;

bool check_cuda() {
    int count;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) return false;
    err = cudaFree(0);
    return err == cudaSuccess;
}

// ── Host data ──────────────────────────────────────────────────────────────

const float SALES[ROWS][COLS] = {
    {12, 11, 4, 3},
    {12,  8, 5, 6},
    { 3,  7, 8, 2}
};

const float FIXED_PRICES[ROWS] = {2, 1, 2};

const float VARYING_PRICES[ROWS][COLS] = {
    {2, 3, 6, 7},
    {1, 8, 3, 7},
    {2, 3, 5, 1}
};

const char* FUELS[ROWS] = {"Diesel", "Gasoline", "Kerosene"};
const char* DAYS[COLS]  = {"Mon", "Tue", "Wed", "Thu"};

// ── CUDA Kernels ───────────────────────────────────────────────────────────

__global__ void fixed_price_kernel(const float* sales, const float* prices,
                                    float* revenue, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int row = idx / cols;
        revenue[idx] = sales[idx] * prices[row];
    }
}

__global__ void hadamard_kernel(const float* A, const float* B,
                                 float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] * B[idx];
    }
}

__global__ void reduction_step_kernel(float* d_A, int n) {
    int mid = n / 2;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < mid) {
        d_A[tid] += d_A[mid + tid];
    }
}

// ── CPU fallback equivalents ───────────────────────────────────────────────

void fixed_price_cpu(const float* sales, const float* prices,
                      float* revenue, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        int row = i / cols;
        revenue[i] = sales[i] * prices[row];
    }
}

void hadamard_cpu(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; i++)
        C[i] = A[i] * B[i];
}

void reduction_step_cpu(float* arr, int n, int* new_len) {
    int mid = n / 2;
    for (int i = 0; i < mid; i++)
        arr[i] += arr[mid + i];
    if (n % 2 == 1)
        arr[mid] = arr[n - 1];
    *new_len = (n + 1) / 2;
}

// ── Host helpers ───────────────────────────────────────────────────────────

int steps_needed(int N) {
    if (N <= 0) return 0;
    int steps = 0;
    while (N > 1) {
        N = (N + 1) / 2;
        steps++;
    }
    return steps;
}

void print_header(const char* title) {
    printf("\n");
    for (int i = 0; i < 70; i++) putchar('=');
    printf("\n %s\n", title);
    for (int i = 0; i < 70; i++) putchar('=');
    printf("\n");
}

void print_array(const float* arr, int n) {
    putchar('[');
    for (int i = 0; i < n; i++) {
        float v = arr[i];
        if (v == (float)(int)v)
            printf("%.0f", v);
        else
            printf("%.1f", v);
        if (i < n - 1) printf(", ");
    }
    putchar(']');
}

void print_matrix(const float* mat, int rows, int cols,
                   const char** row_labels, const char** col_labels, int indent) {
    for (int i = 0; i < indent; i++) putchar(' ');
    printf("%-12s", "");
    for (int c = 0; c < cols; c++)
        printf("%10s", col_labels[c]);
    printf("\n");

    for (int i = 0; i < indent; i++) putchar(' ');
    for (int i = 0; i < 12 + 10 * cols; i++) putchar('-');
    printf("\n");

    for (int r = 0; r < rows; r++) {
        for (int i = 0; i < indent; i++) putchar(' ');
        printf("%-12s", row_labels[r]);
        for (int c = 0; c < cols; c++)
            printf("%10.1f", mat[r * cols + c]);
        printf("\n");
    }
}

// Perform one reduction step and print step details.
//  USE_CUDA=true : d_buf is device, h_buf is host copy
//  USE_CUDA=false: both point to same host buffer
int do_reduction_step_print(float* d_buf, float* h_buf,
                             int n, int buf_capacity, int step_num) {
    int mid = n / 2;
    int t = mid;

    if (USE_CUDA) {
        cudaMemcpy(h_buf, d_buf, buf_capacity * sizeof(float), cudaMemcpyDeviceToHost);
    }

    printf("    Step %d: Left=", step_num);
    print_array(h_buf, t);
    printf(" | Right=");
    print_array(&h_buf[mid], t);
    printf(" | Threads=%d | Result=", t);

    int new_len;
    if (USE_CUDA) {
        int blocks = (t > 0) ? (t + THREADS - 1) / THREADS : 1;
        reduction_step_kernel<<<blocks, THREADS>>>(d_buf, n);
        cudaDeviceSynchronize();

        if (n % 2 == 1) {
            float carry;
            cudaMemcpy(&carry, &d_buf[n - 1], sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&d_buf[mid], &carry, sizeof(float), cudaMemcpyHostToDevice);
        }
        new_len = (n + 1) / 2;
        cudaMemcpy(h_buf, d_buf, buf_capacity * sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        reduction_step_cpu(h_buf, n, &new_len);
    }

    print_array(h_buf, new_len);
    printf("\n");
    return new_len;
}

// Reduce a buffer fully and return the final scalar (no printing).
float reduce_buffer(float* buf, int len, int capacity) {
    if (len <= 0) return 0.0f;

    if (USE_CUDA) {
        float* d_buf = buf;
        float* h_tmp = (float*)malloc(capacity * sizeof(float));

        while (len > 1) {
            int mid = len / 2;
            int t = mid;
            int blocks = (t > 0) ? (t + THREADS - 1) / THREADS : 1;
            reduction_step_kernel<<<blocks, THREADS>>>(d_buf, len);
            cudaDeviceSynchronize();

            if (len % 2 == 1) {
                float carry;
                cudaMemcpy(&carry, &d_buf[len - 1], sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(&d_buf[mid], &carry, sizeof(float), cudaMemcpyHostToDevice);
            }
            len = (len + 1) / 2;
        }

        float result;
        cudaMemcpy(&result, d_buf, sizeof(float), cudaMemcpyDeviceToHost);
        free(h_tmp);
        return result;
    } else {
        while (len > 1) {
            reduction_step_cpu(buf, len, &len);
        }
        return buf[0];
    }
}

// ── Task 1 – Parallel Reduction ───────────────────────────────────────────

void task1() {
    print_header("Task 1 \342\200\223 Parallel Reduction (Sum of Array)");

    float h_A[] = {2, 3, 3, 4, 4, 1, 6, 7, 4, 4};
    int n = 10;

    printf("Initial array: ");
    print_array(h_A, n);
    printf("\n");

    float *d_A = h_A; // used only when USE_CUDA
    if (USE_CUDA) {
        cudaMalloc(&d_A, n * sizeof(float));
        cudaMemcpy(d_A, h_A, n * sizeof(float), cudaMemcpyHostToDevice);
    }

    int current_len = n;
    int step = 0;

    while (current_len > 1) {
        step++;
        if (USE_CUDA) {
            current_len = do_reduction_step_print(d_A, h_A, current_len, n, step);
        } else {
            // h_A is the working buffer — pass same ptr as both d_buf and h_buf
            current_len = do_reduction_step_print(h_A, h_A, current_len, n, step);
        }
    }

    float final_sum = h_A[0];
    if (USE_CUDA) {
        cudaMemcpy(&final_sum, d_A, sizeof(float), cudaMemcpyDeviceToHost);
    }
    printf("Final sum: %.0f\n", final_sum);

    if (USE_CUDA) cudaFree(d_A);

    printf("\nTable of N vs t (steps needed) for N = 1 to 10:\n");
    printf("------------------------------\n");
    printf("%-10s %-10s\n", "N", "t (steps)");
    printf("------------------------------\n");
    for (int N = 1; N <= 10; N++)
        printf("%-10d %-10d\n", N, steps_needed(N));
}

// ── Task 2A – Fixed Price Revenue ─────────────────────────────────────────

void task2a() {
    print_header("Task 2A \342\200\223 Matrix Sales with Fixed Prices");

    float h_revenue[ROWS][COLS];

    if (USE_CUDA) {
        float *d_sales, *d_prices, *d_revenue;
        cudaMalloc(&d_sales,   ROWS * COLS * sizeof(float));
        cudaMalloc(&d_prices,  ROWS * sizeof(float));
        cudaMalloc(&d_revenue, ROWS * COLS * sizeof(float));

        cudaMemcpy(d_sales,  SALES,        ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_prices, FIXED_PRICES, ROWS * sizeof(float),        cudaMemcpyHostToDevice);

        int total = ROWS * COLS;
        int blocks = (total + THREADS - 1) / THREADS;
        fixed_price_kernel<<<blocks, THREADS>>>(d_sales, d_prices, d_revenue, ROWS, COLS);
        cudaDeviceSynchronize();

        cudaMemcpy(h_revenue, d_revenue, ROWS * COLS * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_sales);
        cudaFree(d_prices);
        cudaFree(d_revenue);
    } else {
        fixed_price_cpu((const float*)SALES, FIXED_PRICES, (float*)h_revenue, ROWS, COLS);
    }

    printf("Revenue = Sales \303\227 Fixed Price per fuel\n\n");
    print_matrix((const float*)h_revenue, ROWS, COLS, FUELS, DAYS, 0);
}

// ── Task 2B – Varying Price Revenue (Hadamard) ────────────────────────────

float* task2b() {
    print_header("Task 2B \342\200\223 Matrix Sales with Varying Daily Prices");

    float* h_result = (float*)malloc(ROWS * COLS * sizeof(float));

    if (USE_CUDA) {
        float *d_sales, *d_prices, *d_result;
        cudaMalloc(&d_sales,   ROWS * COLS * sizeof(float));
        cudaMalloc(&d_prices,  ROWS * COLS * sizeof(float));
        cudaMalloc(&d_result,  ROWS * COLS * sizeof(float));

        cudaMemcpy(d_sales,  SALES,          ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_prices, VARYING_PRICES, ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice);

        int total = ROWS * COLS;
        int blocks = (total + THREADS - 1) / THREADS;
        hadamard_kernel<<<blocks, THREADS>>>(d_sales, d_prices, d_result, total);
        cudaDeviceSynchronize();

        cudaMemcpy(h_result, d_result, ROWS * COLS * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_sales);
        cudaFree(d_prices);
        cudaFree(d_result);
    } else {
        hadamard_cpu((const float*)SALES, (const float*)VARYING_PRICES, h_result, ROWS * COLS);
    }

    printf("Revenue = Sales \342\212\231 Varying Prices (Hadamard product)\n\n");
    print_matrix(h_result, ROWS, COLS, FUELS, DAYS, 0);
    return h_result;
}

// ── Task 2C – Total Sales ─────────────────────────────────────────────────

void task2c(float* hadamard) {
    print_header("Task 2C \342\200\223 Total Sales");

    float per_fuel[ROWS];

    if (USE_CUDA) {
        float *d_buf;
        cudaMalloc(&d_buf, ROWS * COLS * sizeof(float));

        for (int f = 0; f < ROWS; f++) {
            cudaMemcpy(d_buf, &hadamard[f * COLS], COLS * sizeof(float), cudaMemcpyHostToDevice);
            per_fuel[f] = reduce_buffer(d_buf, COLS, COLS);
        }

        cudaMemcpy(d_buf, hadamard, ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice);
        float grand_total = reduce_buffer(d_buf, ROWS * COLS, ROWS * COLS);

        cudaFree(d_buf);

        printf("Per-fuel totals:\n");
        for (int f = 0; f < ROWS; f++)
            printf("  %-12s: %.1f\n", FUELS[f], per_fuel[f]);
        printf("\nGrand total: %.1f\n", grand_total);
    } else {
        // Grand total first (before row reductions corrupt data)
        float* copy = (float*)malloc(ROWS * COLS * sizeof(float));
        memcpy(copy, hadamard, ROWS * COLS * sizeof(float));
        int len = ROWS * COLS;
        while (len > 1) {
            reduction_step_cpu(copy, len, &len);
        }
        float grand_total = copy[0];
        free(copy);

        // Row-wise reduction (modifies hadamard in-place)
        for (int f = 0; f < ROWS; f++) {
            float* row = &hadamard[f * COLS];
            int len2 = COLS;
            while (len2 > 1) {
                reduction_step_cpu(row, len2, &len2);
            }
            per_fuel[f] = row[0];
        }

        printf("Per-fuel totals:\n");
        for (int f = 0; f < ROWS; f++)
            printf("  %-12s: %.1f\n", FUELS[f], per_fuel[f]);
        printf("\nGrand total: %.1f\n", grand_total);
    }
}

// ── Task 3 – Simulate CUDA Kernels in Python ──────────────────────────────

void task3() {
    print_header("Task 3 \342\200\223 Simulate CUDA Kernels in Python");

    // Compute Hadamard product
    float h_revenue[ROWS * COLS];

    if (USE_CUDA) {
        float *d_sales, *d_prices, *d_revenue;
        cudaMalloc(&d_sales,   ROWS * COLS * sizeof(float));
        cudaMalloc(&d_prices,  ROWS * COLS * sizeof(float));
        cudaMalloc(&d_revenue, ROWS * COLS * sizeof(float));

        cudaMemcpy(d_sales,  SALES,          ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_prices, VARYING_PRICES, ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice);

        int total = ROWS * COLS;
        int blocks = (total + THREADS - 1) / THREADS;
        hadamard_kernel<<<blocks, THREADS>>>(d_sales, d_prices, d_revenue, total);
        cudaDeviceSynchronize();

        cudaMemcpy(h_revenue, d_revenue, ROWS * COLS * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_sales);
        cudaFree(d_prices);
        cudaFree(d_revenue);
    } else {
        hadamard_cpu((const float*)SALES, (const float*)VARYING_PRICES, h_revenue, ROWS * COLS);
    }

    // Kernel: compute_daily_sales
    printf("\n  \342\226\270 Kernel: compute_daily_sales(S, P)\n");
    printf("  ");
    for (int i = 0; i < 62; i++) putchar('-');
    printf("\n");
    print_matrix(h_revenue, ROWS, COLS, FUELS, DAYS, 2);

    // Kernel: compute_total_sales – block-based verbose reduction
    int block_size = 4;
    int n = ROWS * COLS;

    printf("\n  \342\226\270 Kernel: compute_total_sales(revenue, block_size=%d)\n", block_size);
    printf("  ");
    for (int i = 0; i < 62; i++) putchar('-');
    printf("\n");

    printf("  Flattened array: ");
    print_array(h_revenue, n);
    printf("\n");
    printf("  Length: %d, Block size: %d\n\n", n, block_size);

    int num_blocks = (n + block_size - 1) / block_size;
    float grand_total = 0.0f;

    // Per-block reduction
    for (int b = 0; b < num_blocks; b++) {
        int start = b * block_size;
        int end = (start + block_size < n) ? start + block_size : n;
        int blk_len = end - start;

        printf("  Block %d (indices %2d-%2d): ", b, start, end - 1);
        print_array(&h_revenue[start], blk_len);
        printf("\n");

        // Working buffer for this block
        float block_buf[4]; // max block_size = 4
        memcpy(block_buf, &h_revenue[start], blk_len * sizeof(float));

        float *d_block = block_buf; // used only when USE_CUDA
        if (USE_CUDA) {
            cudaMalloc(&d_block, block_size * sizeof(float));
            cudaMemcpy(d_block, block_buf, blk_len * sizeof(float), cudaMemcpyHostToDevice);
        }

        int current = blk_len;
        int step = 0;

        while (current > 1) {
            step++;
            if (USE_CUDA) {
                current = do_reduction_step_print(d_block, block_buf, current, block_size, step);
            } else {
                current = do_reduction_step_print(block_buf, block_buf, current, block_size, step);
            }
        }

        float partial = block_buf[0];
        if (USE_CUDA) {
            cudaMemcpy(&partial, d_block, sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(d_block);
        }

        printf("    \342\206\222 Block %d partial sum: %.0f\n", b, partial);
        grand_total += partial;
        printf("    \342\206\222 Accumulated total after atomic-add: %.0f\n\n", grand_total);
    }

    printf("  Final accumulated total: %.0f\n", grand_total);
}

// ── Main ──────────────────────────────────────────────────────────────────

int main() {
    USE_CUDA = check_cuda();
    if (USE_CUDA)
        printf("[sales_report] Running on GPU (CUDA)\n\n");
    else
        printf("[sales_report] CUDA not available, using CPU fallback\n\n");

    task1();
    task2a();
    float* hadamard = task2b();
    task2c(hadamard);
    free(hadamard);
    task3();

    printf("\n");
    for (int i = 0; i < 70; i++) putchar('=');
    printf("\n END OF REPORT\n");
    for (int i = 0; i < 70; i++) putchar('=');
    printf("\n\n");

    return 0;
}
