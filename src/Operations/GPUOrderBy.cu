#include "../../include/Operations/GPUOrderBy.cuh"
#include "../../include/Utilities/StringUtils.hpp"
#include "Utilities/ErrorHandling.hpp"
#include <algorithm>
#include <iostream>
#include <cassert>

#define SHARED_MEM_SIZE 1024 // Size can be tuned according to your shared memory availability

// Define a structure that holds sorting column information
struct GPUSortColumn {
    size_t column_index;
    bool is_ascending;
    int type; // Using int instead of enum for simplicity on GPU
};

// Structure for row indices
struct RowIndexValue {
    size_t row_index; // Original row index - needed for stable sorts
};

// CUDA kernel to initialize row indices
__global__ void initRowIndices(RowIndexValue *row_indices, size_t num_rows) {
    size_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx < num_rows) {
        row_indices[row_idx].row_index = row_idx;
    }
}

__device__ int stringCompare(const char *a, const char *b) {
    while (*a != '\0' && *b != '\0') {
        if (*a != *b) {
            return (*a < *b) ? -1 : 1;
        }
        a++;
        b++;
    }
    if (*a == '\0' && *b == '\0') {
        return 0;
    }
    return (*a == '\0') ? -1 : 1;
}

GPUOrderByPlan::GPUOrderByPlan(std::unique_ptr<ExecutionPlan> input,
                               const std::vector<hsql::OrderDescription*>& order_exprs)
    : input_(std::move(input)), order_exprs_(order_exprs) {}

// This function will run on the GPU using shared memory
__device__ bool compareRowIndices(const RowIndexValue &a, const RowIndexValue &b,
                                  const GPUSortColumn *sort_columns, int num_sort_cols, size_t num_cols,
                                  const int *int_data, const double *double_data,
                                  const char *string_data, const size_t *string_offsets) {
    for (int i = 0; i < num_sort_cols; i++) {
        const GPUSortColumn &col = sort_columns[i];
        int cmp = 0;

        switch (col.type) {
        case 1: // INTEGER
            cmp = (int_data[a.row_index * num_cols + col.column_index] <
                   int_data[b.row_index * num_cols + col.column_index]) ? -1 : 
                  (int_data[a.row_index * num_cols + col.column_index] >
                   int_data[b.row_index * num_cols + col.column_index] ? 1 : 0);
            break;
        case 2: // DOUBLE
        case 3: // DATETIME
            cmp = (double_data[a.row_index * num_cols + col.column_index] <
                   double_data[b.row_index * num_cols + col.column_index]) ? -1 :
                  (double_data[a.row_index * num_cols + col.column_index] >
                   double_data[b.row_index * num_cols + col.column_index] ? 1 : 0);
            break;
        case 0: // STRING
            cmp = stringCompare(&string_data[string_offsets[a.row_index * num_cols + col.column_index]],
                                &string_data[string_offsets[b.row_index * num_cols + col.column_index]]);
            break;
        default:
            break;
        }

        if (cmp != 0) {
            return col.is_ascending ? (cmp < 0) : (cmp > 0);
        }
    }

    return a.row_index < b.row_index; // Stable sort
}

// CUDA kernel for bitonic sort step
__global__ void bitonicSortStep(RowIndexValue *row_indices, size_t n,
        unsigned int j, unsigned int k,
        const GPUSortColumn *sort_columns, int num_sort_cols, size_t num_cols,
        const int *int_data, const double *double_data,
        const char *string_data, const size_t *string_offsets) {
    extern __shared__ RowIndexValue shared_row_indices[];

    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
    shared_row_indices[threadIdx.x] = row_indices[i];
    __syncthreads();

    unsigned int ij = i ^ j;
    if (ij >= n) return; // Safeguard return to skip out of bounds processing

    if (ij > i) { // Ensure ij is larger: bitonic property
    bool swap = false;
    if ((i & k) == 0) {
    swap = !compareRowIndices(shared_row_indices[threadIdx.x], shared_row_indices[ij - blockIdx.x * blockDim.x],
                sort_columns, num_sort_cols, num_cols, int_data, double_data, string_data, string_offsets);
    } else {
    swap = compareRowIndices(shared_row_indices[threadIdx.x], shared_row_indices[ij - blockIdx.x * blockDim.x],
                sort_columns, num_sort_cols, num_cols, int_data, double_data, string_data, string_offsets);
    }
    if (swap) {
    RowIndexValue temp = shared_row_indices[threadIdx.x];
    shared_row_indices[threadIdx.x] = shared_row_indices[ij - blockIdx.x * blockDim.x];
    shared_row_indices[ij - blockIdx.x * blockDim.x] = temp;
    }
    }
    __syncthreads();

    row_indices[i] = shared_row_indices[threadIdx.x];
    }
}

std::vector<GPUOrderByPlan::SortColumn> GPUOrderByPlan::parseOrderBy(const Table &table) const {
    std::vector<SortColumn> sort_cols;
    const auto &headers = table.getHeaders();

    for (const auto *order_desc : order_exprs_) {
        if (order_desc->type != hsql::kOrderAsc && order_desc->type != hsql::kOrderDesc)
            throw std::runtime_error("Unsupported ORDER BY type");

        const hsql::Expr *expr = order_desc->expr;
        if (!expr || expr->type != hsql::kExprColumnRef)
            throw std::runtime_error("Only column references are supported in ORDER BY");

        std::string col_name = expr->name;
        size_t col_idx = table.getColumnIndex(col_name);
        ColumnType col_type = table.getColumnType(col_name);
        sort_cols.push_back({col_idx, order_desc->type == hsql::kOrderAsc, col_type});
    }

    return sort_cols;
}

std::shared_ptr<Table> GPUOrderByPlan::execute() {
    std::shared_ptr<Table> table = input_->execute();
    if (!table || table->getData().empty())
        return table;

    std::vector<SortColumn> sort_cols = parseOrderBy(*table);
    if (sort_cols.empty())
        return table;

    const auto &data_map = table->getData();
    const auto &headers = table->getHeaders();
    size_t num_rows = data_map.begin()->second.size();
    size_t num_cols = headers.size();
    size_t pow2_size = 1;
    while (pow2_size < num_rows)
        pow2_size <<= 1;

    std::vector<int> int_data(pow2_size * num_cols, 0);
    std::vector<double> double_data(pow2_size * num_cols, 0.0);
    std::vector<char> string_data;
    std::vector<size_t> string_offsets(pow2_size * num_cols, 0);
    size_t string_data_offset = 0;

    // Initialize data buffers
    for (size_t row_idx = 0; row_idx < num_rows; ++row_idx) {
        for (size_t col_idx = 0; col_idx < num_cols; ++col_idx) {
            const std::string &col_name = headers[col_idx];
            ColumnType col_type = table->getColumnType(col_name);
            const unionV &val = data_map.at(col_name)[row_idx];
            size_t flat_idx = row_idx * num_cols + col_idx;

            switch (col_type) {
            case ColumnType::INTEGER:
                int_data[flat_idx] = val.i;
                break;
            case ColumnType::DOUBLE:
            case ColumnType::DATETIME:
                double_data[flat_idx] = val.d; // Treat DATETIME as double
                break;
            case ColumnType::STRING:
                if (val.s != nullptr) {
                    std::string str = *val.s;
                    string_offsets[flat_idx] = string_data_offset;
                    string_data.insert(string_data.end(), str.c_str(), str.c_str() + str.length() + 1);
                    string_data_offset += str.length() + 1;
                } else {
                    throw SemanticError("Null string value encountered during GPU ORDER BY processing.");
                }
                break;
            default:
                throw SemanticError("Unsupported column type in GPU ORDER BY");
            }
        }
    }

    // Populate buffers for remaining rows with zeros
    for (size_t row_idx = num_rows; row_idx < pow2_size; ++row_idx) {
        for (size_t col_idx = 0; col_idx < num_cols; ++col_idx) {
            size_t flat_idx = row_idx * num_cols + col_idx;
            string_offsets[flat_idx] = string_data_offset;
        }
    }

    std::vector<GPUSortColumn> gpu_sort_cols(sort_cols.size());
    for (size_t i = 0; i < sort_cols.size(); ++i) {
        gpu_sort_cols[i] = {sort_cols[i].column_index, sort_cols[i].is_ascending, static_cast<int>(sort_cols[i].type)};
    }

    int *d_int_data = nullptr;
    double *d_double_data = nullptr;
    char *d_string_data = nullptr;
    size_t *d_string_offsets = nullptr;
    GPUSortColumn *d_sort_cols = nullptr;
    RowIndexValue *d_row_indices = nullptr;

    // Allocate and handle GPU operations
    try {
        CUDA_CHECK(cudaMalloc(&d_int_data, sizeof(int) * num_rows * num_cols));
        CUDA_CHECK(cudaMalloc(&d_double_data, sizeof(double) * num_rows * num_cols));
        CUDA_CHECK(cudaMalloc(&d_string_data, sizeof(char) * string_data.size()));
        CUDA_CHECK(cudaMalloc(&d_string_offsets, sizeof(size_t) * num_rows * num_cols));
        CUDA_CHECK(cudaMalloc(&d_sort_cols, sizeof(GPUSortColumn) * gpu_sort_cols.size()));
        CUDA_CHECK(cudaMalloc(&d_row_indices, sizeof(RowIndexValue) * pow2_size));

        CUDA_CHECK(cudaMemcpy(d_int_data, int_data.data(), sizeof(int) * int_data.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_double_data, double_data.data(), sizeof(double) * double_data.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_string_data, string_data.data(), sizeof(char) * string_data.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_string_offsets, string_offsets.data(), sizeof(size_t) * string_offsets.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sort_cols, gpu_sort_cols.data(), sizeof(GPUSortColumn) * gpu_sort_cols.size(), cudaMemcpyHostToDevice));

        int threadsPerBlock = 256;
        int blocksPerGrid = (pow2_size + threadsPerBlock - 1) / threadsPerBlock;

        initRowIndices<<<blocksPerGrid, threadsPerBlock>>>(d_row_indices, pow2_size);
        CUDA_CHECK(cudaGetLastError());

        size_t shared_mem_size = sizeof(RowIndexValue) * threadsPerBlock;
        for (unsigned int k = 2; k <= pow2_size; k <<= 1) {
            for (unsigned int j = k >> 1; j > 0; j >>= 1) {
                bitonicSortStep<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(
                    d_row_indices, pow2_size, j, k, d_sort_cols, gpu_sort_cols.size(), num_cols,
                    d_int_data, d_double_data, d_string_data, d_string_offsets);
                CUDA_CHECK(cudaGetLastError());
            }
        }

        std::vector<RowIndexValue> sorted_row_indices(pow2_size);
        CUDA_CHECK(cudaMemcpy(sorted_row_indices.data(), d_row_indices, sizeof(RowIndexValue) * pow2_size, cudaMemcpyDeviceToHost));

        std::unordered_map<std::string, std::vector<unionV>> sorted_data_map;
        for (const auto &header : headers) {
            sorted_data_map[header].reserve(num_rows);
        }
        for (size_t i = 0; i < num_rows; ++i) {
            size_t orig_row_idx = sorted_row_indices[i].row_index;
            for (const auto &header : headers) {
                sorted_data_map[header].push_back(data_map.at(header)[orig_row_idx]);
            }
        }

        CUDA_CHECK(cudaFree(d_int_data));
        CUDA_CHECK(cudaFree(d_double_data));
        CUDA_CHECK(cudaFree(d_string_data));
        CUDA_CHECK(cudaFree(d_string_offsets));
        CUDA_CHECK(cudaFree(d_sort_cols));
        CUDA_CHECK(cudaFree(d_row_indices));

        return std::make_shared<Table>(
            table->getName() + "_ordered",
            headers,
            std::move(sorted_data_map),
            table->getColumnTypes());
    } catch (const std::exception &e) {
        if (d_int_data) cudaFree(d_int_data);
        if (d_double_data) cudaFree(d_double_data);
        if (d_string_data) cudaFree(d_string_data);
        if (d_string_offsets) cudaFree(d_string_offsets);
        if (d_sort_cols) cudaFree(d_sort_cols);
        if (d_row_indices) cudaFree(d_row_indices);
        throw; // Propagate error
    }
}