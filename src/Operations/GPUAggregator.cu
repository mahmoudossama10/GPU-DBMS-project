// #include "../../include/Operations/GPUAggregator.hpp"
// #include <cuda_runtime.h>
// #include <vector>
// #include <climits>
// #include <stdexcept>
// #include <cstring>
// #include <algorithm>
// #include <limits>
// #include <numeric>

// // ------------------ Kernel ------------------


// struct IntAggregatesDevice {

//     long long sum;

//     int count;

//     int min;

//     int max;

// };


// __global__

// void multiAggregate_kernel(const int* data, int N, IntAggregatesDevice* result) {

//     extern __shared__ int buf[]; // blockDim.x ints for reductions

//     int tid = threadIdx.x;

//     int globalIdx = blockIdx.x * blockDim.x + tid;


//     // Local partials

//     long long tsum = 0;

//     int tmin = INT_MAX, tmax = INT_MIN, tcount = 0;


//     // Stride through the array (to support any N)

//     for (int i = globalIdx; i < N; i += blockDim.x * gridDim.x) {

//         int val = data[i];

//         tsum += val;

//         tmin = min(tmin, val);

//         tmax = max(tmax, val);

//         ++tcount;

//     }


//     // Reduce within the block

//     // use shared memory for sum/min/max/count

//     __shared__ long long s_sum[256];

//     __shared__ int s_min[256], s_max[256], s_count[256];

//     s_sum[tid] = tsum;

//     s_min[tid] = tmin;

//     s_max[tid] = tmax;

//     s_count[tid] = tcount;


//     __syncthreads();


//     for (int s = blockDim.x/2; s>0; s>>=1) {

//         if (tid < s) {

//             s_sum[tid] += s_sum[tid+s];

//             s_min[tid] = min(s_min[tid], s_min[tid+s]);

//             s_max[tid] = max(s_max[tid], s_max[tid+s]);

//             s_count[tid] += s_count[tid+s];

//         }

//         __syncthreads();

//     }


//     // Block leader atomically adds results to global result

//     if (tid == 0) {

//         atomicAdd((unsigned long long*)&(result->sum), (unsigned long long)s_sum[0]);

//         atomicMin(&(result->min), s_min[0]);

//         atomicMax(&(result->max), s_max[0]);

//         atomicAdd(&(result->count), s_count[0]);

//     }

// }


// // Host wrapper: Batched aggregation

// GPUAggregator::IntAggregates GPUAggregator::multiAggregateInt(const std::vector<int>& col) {

//     IntAggregates out;

//     if(col.empty()) return out;


//     int N = col.size();

//     int* d_col = nullptr;

//     IntAggregatesDevice* d_result = nullptr;

//     IntAggregatesDevice h_result;


//     h_result.sum = 0;

//     h_result.count = 0;

//     h_result.min = INT_MAX;

//     h_result.max = INT_MIN;


//     cudaMalloc(&d_col, N*sizeof(int));

//     cudaMemcpy(d_col, col.data(), N*sizeof(int), cudaMemcpyHostToDevice);

//     cudaMalloc(&d_result, sizeof(IntAggregatesDevice));

//     cudaMemcpy(d_result, &h_result, sizeof(IntAggregatesDevice), cudaMemcpyHostToDevice);


//     int threads = 256;

//     int blocks = (N + threads - 1) / threads;

//     if (blocks > 1024) blocks = 1024;


//     multiAggregate_kernel<<<blocks, threads, 0>>>(d_col, N, d_result);

//     cudaDeviceSynchronize();


//     cudaMemcpy(&h_result, d_result, sizeof(IntAggregatesDevice), cudaMemcpyDeviceToHost);


//     cudaFree(d_col);

//     cudaFree(d_result);


//     out.sum = h_result.sum;

//     out.count = h_result.count;

//     out.min = h_result.min;

//     out.max = h_result.max;

//     out.avg = (h_result.count > 0) ? double(h_result.sum) / h_result.count : 0.0;


//     return out;

// }

// // CUDA error helper for debugging
// static inline void checkCuda(cudaError_t result, char const *const func, const char *const file, int const line) {
//     if (result != cudaSuccess) {
//         fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
//                 file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
//         exit(EXIT_FAILURE);
//     }
// }
// #define checkCudaErrors(val) checkCuda((val), #val, __FILE__, __LINE__)

// // -------------------------------------------
// // REDUCTION KERNELS
// // opType: 0=SUM, 1=MIN, 2=MAX
// // Device version

// __device__ int reduce_op_device(int a, int b, int op) {
//     if(op==0) return a+b;
//     if(op==1) return (a<b)?a:b;
//     if(op==2) return (a>b)?a:b;
//     return 0;
// }

// // Host version
// inline int reduce_op_host(int a, int b, int op) {
//     if(op==0) return a+b;
//     if(op==1) return (a<b)?a:b;
//     if(op==2) return (a>b)?a:b;
//     return 0;
// }

// __global__ void reduce_kernel(const int* data, int* partial, int n, int opType, int identity) {
//     extern __shared__ int sdata[];
//     int tid = threadIdx.x;
//     int i = blockIdx.x * blockDim.x * 2 + tid;
//     int myval = identity;
//     if (i < n) myval = data[i];
//     if (i + blockDim.x < n) myval = reduce_op_device(myval, data[i + blockDim.x], opType);
//     sdata[tid] = myval;
//     __syncthreads();

//     for (int s = blockDim.x/2; s>0; s>>=1) {
//         if (tid < s)
//             sdata[tid] = reduce_op_device(sdata[tid], sdata[tid+s], opType);
//         __syncthreads();
//     }
//     if (tid==0) partial[blockIdx.x] = sdata[0];
// }

// // Host-side reduction wrapper (for int)
// static int gpu_reduce(const std::vector<int>& col, int opType) {
//     int N = col.size();
//     if (N == 0)
//         throw std::runtime_error("Cannot reduce empty vector.");

//     int identity = 0;
//     if(opType==1) identity = INT_MAX; // min
//     if(opType==2) identity = INT_MIN; // max

//     int* d_in = nullptr;
//     int* d_out = nullptr;
//     int threads = 256;
//     int blocks = (N + threads*2 - 1) / (threads*2);
//     checkCudaErrors(cudaMalloc(&d_in, N * sizeof(int)));
//     checkCudaErrors(cudaMemcpy(d_in, col.data(), N * sizeof(int), cudaMemcpyHostToDevice));
//     checkCudaErrors(cudaMalloc(&d_out, blocks * sizeof(int)));
//     reduce_kernel<<<blocks, threads, threads * sizeof(int)>>>(d_in, d_out, N, opType, identity);
//     checkCudaErrors(cudaDeviceSynchronize());
//     std::vector<int> h_partial(blocks);
//     checkCudaErrors(cudaMemcpy(h_partial.data(), d_out, blocks * sizeof(int), cudaMemcpyDeviceToHost));
//     cudaFree(d_in);
//     cudaFree(d_out);
//     int result = identity;
//     for(int i=0; i<blocks; ++i) result = reduce_op_host(result, h_partial[i], opType); // use host version here!
//     return result;

// }

// int GPUAggregator::sumInt(const std::vector<int>& col)     { return gpu_reduce(col, 0); }
// int GPUAggregator::minInt(const std::vector<int>& col)     { return gpu_reduce(col, 1); }
// int GPUAggregator::maxInt(const std::vector<int>& col)     { return gpu_reduce(col, 2); }
// int GPUAggregator::countInt(const std::vector<int>& col)   { return int(col.size()); }
// double GPUAggregator::avgInt(const std::vector<int>& col)  { return (col.empty() ? std::numeric_limits<double>::quiet_NaN() : static_cast<double>(gpu_reduce(col, 0)) / col.size()); }

// // -------------------------------------------
// // ARGSORT KERNEL (Bitonic, for small-medium N)

// __global__ void bitonic_argsort_kernel(int* data, int* indices, int n, int ascending) {
//     // Shared mem for block (N ints, N indices)
//     extern __shared__ int shared[];
//     int* arr = shared;
//     int* idx = shared + n;
//     int tid = threadIdx.x;
//     // Init
//     if (tid < n) {
//         arr[tid] = data[tid];
//         idx[tid] = tid;
//     }
//     __syncthreads();

//     // Bitonic sort, works best for power-of-2 N, but still correct otherwise for small N
//     for (int k = 2; k <= n; k <<= 1) {
//         for (int j = k >> 1; j > 0; j >>= 1) {
//             int ixj = tid ^ j;
//             if (ixj > tid && tid < n) {
//                 // Compare for ascending/descending
//                 bool swap = false;
//                 if ( ((tid & k) == 0 && ((arr[tid] > arr[ixj]) == ascending)) ||
//                     ((tid & k) != 0 && ((arr[tid] < arr[ixj]) == ascending)) )
//                 {
//                     swap = true;
//                 }
//                 if (swap) {
//                     int tmpv = arr[tid];
//                     arr[tid] = arr[ixj];
//                     arr[ixj] = tmpv;
//                     int tmpi = idx[tid];
//                     idx[tid] = idx[ixj];
//                     idx[ixj] = tmpi;
//                 }
//             }
//             __syncthreads();
//         }
//     }
//     if(tid < n)
//         indices[tid] = idx[tid];
// }

// // Host wrapper for ARGSORT
// std::vector<int> GPUAggregator::gpuArgsortInt(const std::vector<int>& col, bool ascending) {
//     int N = col.size();
//     if (N == 0) return {};

//     // Use bitonic only up to 8192 rows
//     const int MAX_N = 8192;
//     if (N > MAX_N) {
//         // Fallback to fast CPU std::argsort for huge N
//         std::vector<int> indices(N);
//         std::iota(indices.begin(), indices.end(), 0);
//         std::sort(indices.begin(), indices.end(), [&](int a, int b){
//             return ascending ? col[a] < col[b] : col[a] > col[b];
//         });
//         return indices;
//     }

//     int* d_col = nullptr;
//     int* d_idx = nullptr;
//     checkCudaErrors(cudaMalloc(&d_col, N * sizeof(int)));
//     checkCudaErrors(cudaMalloc(&d_idx, N * sizeof(int)));
//     checkCudaErrors(cudaMemcpy(d_col, col.data(), N * sizeof(int), cudaMemcpyHostToDevice));

//     int blockSize = 1;
//     int threadsPerBlock = N;
//     int smemSize = 2 * N * sizeof(int);

//     // Kernel launch safely — 1 block, N threads, enough shared mem
//     bitonic_argsort_kernel<<<blockSize, threadsPerBlock, smemSize>>>(d_col, d_idx, N, ascending ? 1 : 0);
//     checkCudaErrors(cudaDeviceSynchronize());

//     std::vector<int> host_idx(N);
//     checkCudaErrors(cudaMemcpy(host_idx.data(), d_idx, N * sizeof(int), cudaMemcpyDeviceToHost));

//     cudaFree(d_col);
//     cudaFree(d_idx);

//     return host_idx;
// }