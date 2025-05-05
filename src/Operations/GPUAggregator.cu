#include "../../include/Operations/GPUAggregator.cuh"
#include "Utilities/ErrorHandling.hpp"
#include <algorithm>
#include <unordered_set>
#include <cuda_runtime.h>
#include <climits>
#include <cfloat>

#define CUDA_CHECK(err) do { \
    cudaError_t err_val = (err); \
    if (err_val != cudaSuccess) { \
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err_val))); \
    } \
} while(0)

#define THREADS_PER_BLOCK 256

// CUDA kernel for COUNT operation
__global__ void countKernel(size_t num_rows, int* result) {
    extern __shared__ int shared_count[];
    int tid = threadIdx.x;
    shared_count[tid] = 0;

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_rows) {
        shared_count[tid] = 1; // Each thread counts 1 for each valid row
    }
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_count[tid] += shared_count[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, shared_count[0]); // atomicAdd is supported for int
    }
}

// CUDA kernel for SUM operation (for double)
__global__ void sumKernelDouble(const double* data, size_t num_rows, double* result) {
    extern __shared__ double shared_sum_double[];
    int tid = threadIdx.x;
    shared_sum_double[tid] = 0.0;

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_rows) {
        shared_sum_double[tid] = data[idx];
    }
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum_double[tid] += shared_sum_double[tid + s];
        }
        __syncthreads();
    }

    // Single thread updates the global result to avoid atomicAdd issues with double
    if (tid == 0) {
        *result += shared_sum_double[0];
    }
}

// CUDA kernel for SUM operation (for int64_t)
__global__ void sumKernelInt(const int64_t* data, size_t num_rows, int64_t* result) {
    extern __shared__ int64_t shared_sum_int[];
    int tid = threadIdx.x;
    shared_sum_int[tid] = 0;

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_rows) {
        shared_sum_int[tid] = data[idx];
    }
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum_int[tid] += shared_sum_int[tid + s];
        }
        __syncthreads();
    }

    // Single thread updates the global result to avoid atomicAdd issues with int64_t on older architectures
    if (tid == 0) {
        *result += shared_sum_int[0];
    }
}

// CUDA kernel for MIN/MAX operation (for double)
__global__ void minMaxKernelDouble(const double* data, size_t num_rows, double* result, bool is_min) {
    extern __shared__ double shared_val_double[];
    int tid = threadIdx.x;
    shared_val_double[tid] = is_min ? 1.0e308 : -1.0e308; // Large/small value for min/max initialization

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_rows) {
        shared_val_double[tid] = data[idx];
    }
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (is_min) {
                shared_val_double[tid] = (shared_val_double[tid] < shared_val_double[tid + s]) ? shared_val_double[tid] : shared_val_double[tid + s];
            } else {
                shared_val_double[tid] = (shared_val_double[tid] > shared_val_double[tid + s]) ? shared_val_double[tid] : shared_val_double[tid + s];
            }
        }
        __syncthreads();
    }

    // Single thread updates the global result to avoid atomicMin/Max issues with double
    if (tid == 0) {
        double current = *result;
        if (is_min) {
            if (shared_val_double[0] < current) *result = shared_val_double[0];
        } else {
            if (shared_val_double[0] > current) *result = shared_val_double[0];
        }
    }
}

// CUDA kernel for MIN/MAX operation (for int64_t)
__global__ void minMaxKernelInt(const int64_t* data, size_t num_rows, int64_t* result, bool is_min) {
    extern __shared__ int64_t shared_val_int[];
    int tid = threadIdx.x;
    shared_val_int[tid] = is_min ? LLONG_MAX : LLONG_MIN;

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_rows) {
        shared_val_int[tid] = data[idx];
    }
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (is_min) {
                shared_val_int[tid] = (shared_val_int[tid] < shared_val_int[tid + s]) ? shared_val_int[tid] : shared_val_int[tid + s];
            } else {
                shared_val_int[tid] = (shared_val_int[tid] > shared_val_int[tid + s]) ? shared_val_int[tid] : shared_val_int[tid + s];
            }
        }
        __syncthreads();
    }

    // Single thread updates the global result to avoid atomicMin/Max issues with int64_t on older architectures
    if (tid == 0) {
        int64_t current = *result;
        if (is_min) {
            if (shared_val_int[0] < current) *result = shared_val_int[0];
        } else {
            if (shared_val_int[0] > current) *result = shared_val_int[0];
        }
    }
}

GPUAggregatorPlan::GPUAggregatorPlan(std::unique_ptr<ExecutionPlan> input, const std::vector<hsql::Expr*>& select_list)
    : input_(std::move(input)), select_list_(select_list) {}

std::shared_ptr<Table> GPUAggregatorPlan::execute() {
    std::shared_ptr<Table> table = input_->execute();
    if (!table || table->getData().empty()) {
        return table;
    }

    auto aggregates = parseAggregates(select_list_, *table);
    return aggregateTableGPU(*table, aggregates);
}

std::vector<GPUAggregatorPlan::AggregateOp> GPUAggregatorPlan::parseAggregates(
    const std::vector<hsql::Expr*>& select_list, const Table& table) {
    std::vector<AggregateOp> aggregates;

    for (const auto* expr : select_list) {
        if (expr->type == hsql::kExprFunctionRef && expr->name) {
            std::string func_name = expr->name;
            std::transform(func_name.begin(), func_name.end(), func_name.begin(), ::tolower);

            if (func_name == "count" || func_name == "sum" || func_name == "avg" ||
                func_name == "min" || func_name == "max") {
                AggregateOp op;
                op.function_name = func_name;
                op.is_distinct = expr->distinct;

                if (expr->exprList && !expr->exprList->empty()) {
                    const auto* arg = expr->exprList->at(0);
                    if (arg->type == hsql::kExprColumnRef && arg->name) {
                        op.column_name = arg->name;
                        if (!table.hasColumn(op.column_name)) {
                            throw SemanticError("Column not found for aggregate: " + op.column_name);
                        }
                        op.column_index = table.getColumnIndex(op.column_name);
                    } else if (arg->type == hsql::kExprStar && func_name == "count") {
                        op.column_name = table.getHeaders()[0];
                        op.column_index = 0;
                    } else {
                        throw SemanticError("Invalid argument for aggregate function: " + func_name);
                    }
                } else {
                    throw SemanticError("No arguments provided for aggregate function: " + func_name);
                }

                op.alias = expr->alias ? expr->alias : func_name + "(" + op.column_name + ")";
                aggregates.push_back(op);
            } else {
                throw SemanticError("Unsupported aggregate function: " + func_name);
            }
        }
    }

    return aggregates;
}

// Helper function to convert unionV to string for distinct operations
std::string unionValueToString(const unionV& value, ColumnType type) {
    switch (type) {
    case ColumnType::STRING:
        return *(value.s);
    case ColumnType::INTEGER:
        return std::to_string(value.i);
    case ColumnType::DOUBLE:
        return std::to_string(value.d);
    case ColumnType::DATETIME: {
        char buffer[64];
        snprintf(buffer, sizeof(buffer), "%04hu-%02hu-%02hu %02hhu:%02hhu:%02hhu",
                 value.t->year, value.t->month, value.t->day,
                 value.t->hour, value.t->minute, value.t->second);
        return std::string(buffer);
    }
    default:
        throw std::runtime_error("Unknown column type");
    }
}

std::shared_ptr<Table> GPUAggregatorPlan::aggregateTableGPU(
    const Table& table, const std::vector<AggregateOp>& aggregates) {
    if (aggregates.empty()) {
        throw SemanticError("No aggregate operations to perform");
    }

    std::unordered_map<std::string, std::vector<unionV>> result_data;
    std::vector<std::string> result_headers;
    std::unordered_map<std::string, ColumnType> result_types;
    size_t num_rows = table.getSize();

    for (const auto& op : aggregates) {
        result_headers.push_back(op.alias);
        ColumnType col_type = table.getColumnType(op.column_name);
        result_types[op.alias] = (op.function_name == "count") ? ColumnType::INTEGER : col_type;

        std::vector<unionV> result_col(1);

        if (col_type == ColumnType::STRING && op.function_name != "count") {
            if (op.function_name == "min" || op.function_name == "max") {
                std::string extreme = (op.function_name == "min") ? table.getString(op.column_name, 0) : table.getString(op.column_name, 0);
                for (size_t i = 1; i < num_rows; ++i) {
                    std::string val = table.getString(op.column_name, i);
                    if (op.function_name == "min" ? val < extreme : val > extreme) {
                        extreme = val;
                    }
                }
                result_col[0].s = new std::string(extreme);
            } else {
                throw SemanticError("Unsupported aggregate operation on STRING type: " + op.function_name);
            }
        } else {
            if (op.function_name == "count") {
                int h_result = 0;
                int* d_result = nullptr;
                CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));
                CUDA_CHECK(cudaMemset(d_result, 0, sizeof(int)));

                int blocks = (num_rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                size_t shared_mem_size = THREADS_PER_BLOCK * sizeof(int);
                countKernel<<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(num_rows, d_result);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());

                CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaFree(d_result));

                if (op.is_distinct) {
                    std::unordered_set<std::string> unique_values;
                    for (size_t i = 0; i < num_rows; ++i) {
                        std::string val_str;
                        switch (col_type) {
                            case ColumnType::STRING:
                                val_str = table.getString(op.column_name, i);
                                break;
                            case ColumnType::INTEGER:
                                val_str = std::to_string(table.getInteger(op.column_name, i));
                                break;
                            case ColumnType::DOUBLE:
                                val_str = std::to_string(table.getDouble(op.column_name, i));
                                break;
                            case ColumnType::DATETIME:
                                val_str = unionValueToString(table.getRow(i)[op.column_index], col_type);
                                break;
                        }
                        unique_values.insert(val_str);
                    }
                    result_col[0].i = unique_values.size();
                } else {
                    result_col[0].i = h_result;
                }
            } else if (op.function_name == "sum" || op.function_name == "avg") {
                if (col_type == ColumnType::INTEGER) {
                    std::vector<int64_t> h_data(num_rows);
                    for (size_t i = 0; i < num_rows; ++i) {
                        h_data[i] = table.getInteger(op.column_name, i);
                    }

                    int64_t h_result = 0;
                    int64_t* d_data = nullptr;
                    int64_t* d_result = nullptr;
                    CUDA_CHECK(cudaMalloc(&d_data, num_rows * sizeof(int64_t)));
                    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int64_t)));
                    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(int64_t)));
                    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), num_rows * sizeof(int64_t), cudaMemcpyHostToDevice));

                    int blocks = (num_rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                    size_t shared_mem_size = THREADS_PER_BLOCK * sizeof(int64_t);
                    sumKernelInt<<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(d_data, num_rows, d_result);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaDeviceSynchronize());

                    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int64_t), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaFree(d_data));
                    CUDA_CHECK(cudaFree(d_result));

                    if (op.function_name == "avg") {
                        result_col[0].d = static_cast<double>(h_result) / num_rows;
                    } else {
                        result_col[0].i = h_result;
                    }
                } else if (col_type == ColumnType::DOUBLE || col_type == ColumnType::DATETIME) {
                    std::vector<double> h_data(num_rows);
                    for (size_t i = 0; i < num_rows; ++i) {
                        h_data[i] = (col_type == ColumnType::DOUBLE) ? table.getDouble(op.column_name, i) : 0.0;
                    }

                    double h_result = 0.0;
                    double* d_data = nullptr;
                    double* d_result = nullptr;
                    CUDA_CHECK(cudaMalloc(&d_data, num_rows * sizeof(double)));
                    CUDA_CHECK(cudaMalloc(&d_result, sizeof(double)));
                    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(double)));
                    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), num_rows * sizeof(double), cudaMemcpyHostToDevice));

                    int blocks = (num_rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                    size_t shared_mem_size = THREADS_PER_BLOCK * sizeof(double);
                    sumKernelDouble<<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(d_data, num_rows, d_result);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaDeviceSynchronize());

                    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaFree(d_data));
                    CUDA_CHECK(cudaFree(d_result));

                    if (op.function_name == "avg") {
                        result_col[0].d = h_result / num_rows;
                    } else {
                        result_col[0].d = h_result;
                    }
                }
            } else if (op.function_name == "min" || op.function_name == "max") {
                bool is_min = (op.function_name == "min");
                if (col_type == ColumnType::INTEGER) {
                    std::vector<int64_t> h_data(num_rows);
                    for (size_t i = 0; i < num_rows; ++i) {
                        h_data[i] = table.getInteger(op.column_name, i);
                    }

                    int64_t h_result = is_min ? LLONG_MAX : LLONG_MIN;
                    int64_t* d_data = nullptr;
                    int64_t* d_result = nullptr;
                    CUDA_CHECK(cudaMalloc(&d_data, num_rows * sizeof(int64_t)));
                    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int64_t)));
                    CUDA_CHECK(cudaMemcpy(d_result, &h_result, sizeof(int64_t), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), num_rows * sizeof(int64_t), cudaMemcpyHostToDevice));

                    int blocks = (num_rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                    size_t shared_mem_size = THREADS_PER_BLOCK * sizeof(int64_t);
                    minMaxKernelInt<<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(d_data, num_rows, d_result, is_min);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaDeviceSynchronize());

                    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int64_t), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaFree(d_data));
                    CUDA_CHECK(cudaFree(d_result));

                    result_col[0].i = h_result;
                } else if (col_type == ColumnType::DOUBLE || col_type == ColumnType::DATETIME) {
                    std::vector<double> h_data(num_rows);
                    for (size_t i = 0; i < num_rows; ++i) {
                        h_data[i] = (col_type == ColumnType::DOUBLE) ? table.getDouble(op.column_name, i) : 0.0;
                    }

                    double h_result = is_min ? 1.0e308 : -1.0e308;
                    double* d_data = nullptr;
                    double* d_result = nullptr;
                    CUDA_CHECK(cudaMalloc(&d_data, num_rows * sizeof(double)));
                    CUDA_CHECK(cudaMalloc(&d_result, sizeof(double)));
                    CUDA_CHECK(cudaMemcpy(d_result, &h_result, sizeof(double), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), num_rows * sizeof(double), cudaMemcpyHostToDevice));

                    int blocks = (num_rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                    size_t shared_mem_size = THREADS_PER_BLOCK * sizeof(double);
                    minMaxKernelDouble<<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(d_data, num_rows, d_result, is_min);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaDeviceSynchronize());

                    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaFree(d_data));
                    CUDA_CHECK(cudaFree(d_result));

                    result_col[0].d = h_result;
                }
            }
        }
        result_data[op.alias] = std::move(result_col);
    }

    return std::make_shared<Table>(
        table.getName() + "_gpu_aggregated",
        result_headers,
        std::move(result_data),
        result_types);
}