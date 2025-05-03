#include "../../include/Operations/GPUOrderBy.cuh"
#include "../../include/Utilities/StringUtils.hpp"
#include "Utilities/ErrorHandling.hpp"
#include <algorithm>
#include <iostream>

GPUOrderByPlan::GPUOrderByPlan(std::unique_ptr<ExecutionPlan> input,
                               const std::vector<hsql::OrderDescription *> &order_exprs)
    : input_(std::move(input)), order_exprs_(order_exprs) {}

// Define a structure that can be passed to the GPU
struct GPUSortColumn
{
    size_t column_index;
    bool is_ascending;
    int type; // Using int instead of enum for simplicity on GPU
};

// Structure for row indices
struct RowIndexValue
{
    size_t row_index; // Original row index - needed for stable sort
};

// CUDA kernel for preparing row indices
__global__ void prepareRowIndices(const int *int_data, const double *double_data,
                                  const char *string_data, size_t *string_offsets,
                                  RowIndexValue *row_indices, size_t num_rows, size_t num_cols,
                                  GPUSortColumn *sort_columns, int num_sort_columns,
                                  int primary_sort_column_index)
{
    size_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= num_rows)
        return;

    // Set the row index
    row_indices[row_idx].row_index = row_idx;
}

// CUDA comparison function for bitonic sort - STABLE version

__device__ bool compareRowIndices(const RowIndexValue &a, const RowIndexValue &b,
                                  const GPUSortColumn *sort_columns, int num_sort_cols, size_t num_cols, size_t original_num_rows,
                                  const int *int_data, const double *double_data,
                                  const char *string_data, size_t *string_offsets)
{
    if (a.row_index >= original_num_rows && b.row_index < original_num_rows)
    {
        return false; // a > b
    }
    if (a.row_index < original_num_rows && b.row_index >= original_num_rows)
    {
        return true; // a < b
    }
    if (a.row_index >= original_num_rows && b.row_index >= original_num_rows)
    {
        return a.row_index < b.row_index;
    }

    for (int i = 0; i < num_sort_cols; i++)
    {
        const GPUSortColumn &col = sort_columns[i];
        int cmp = 0;
        switch (col.type)
        {
        case 1: // INTEGER
        {
            int val_a = int_data[a.row_index * num_cols + col.column_index];
            int val_b = int_data[b.row_index * num_cols + col.column_index];
            cmp = (val_a < val_b) ? -1 : (val_a > val_b ? 1 : 0);
        }
        break;
        case 2: // DOUBLE
        {
            double val_a = double_data[a.row_index * num_cols + col.column_index];
            double val_b = double_data[b.row_index * num_cols + col.column_index];
            cmp = (val_a < val_b) ? -1 : (val_a > val_b ? 1 : 0);
        }
        break;
        case 3: // DATETIME
        {
            double val_a = double_data[a.row_index * num_cols + col.column_index];
            double val_b = double_data[b.row_index * num_cols + col.column_index];
            cmp = (val_a < val_b) ? -1 : (val_a > val_b ? 1 : 0);
        }
        break;
        case 0: // STRING
        {
            size_t offset_a = string_offsets[a.row_index * num_cols + col.column_index];
            size_t offset_b = string_offsets[b.row_index * num_cols + col.column_index];
            int j = 0;
            while (true)
            {
                char char_a = string_data[offset_a + j];
                char char_b = string_data[offset_b + j];
                if (char_a != char_b)
                {
                    cmp = char_a - char_b;
                    break;
                }
                if (char_a == '\0')
                {
                    cmp = 0;
                    break;
                }
                j++;
            }
        }
        break;
        default:
            break;
        }

        if (cmp != 0)
        {
            return col.is_ascending ? (cmp < 0) : (cmp > 0);
        }
    }

    return a.row_index < b.row_index; // Stable sort
}

// CUDA kernel for bitonic sort step
__global__ void bitonicSortStep(RowIndexValue *row_indices, size_t n,
                                unsigned int j, unsigned int k,
                                const GPUSortColumn *sort_columns, int num_sort_cols, size_t num_cols, size_t original_num_rows,
                                const int *int_data, const double *double_data,
                                const char *string_data, size_t *string_offsets)
{
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n)
        return;

    unsigned int ij = i ^ j;
    if (ij > i)
    {
        bool swap = false;
        if ((i & k) == 0)
        {
            swap = !compareRowIndices(row_indices[i], row_indices[ij],
                                      sort_columns, num_sort_cols, num_cols, original_num_rows,
                                      int_data, double_data, string_data, string_offsets);
        }
        else
        {
            swap = compareRowIndices(row_indices[i], row_indices[ij],
                                     sort_columns, num_sort_cols, num_cols, original_num_rows,
                                     int_data, double_data, string_data, string_offsets);
        }
        if (swap)
        {
            RowIndexValue temp = row_indices[i];
            row_indices[i] = row_indices[ij];
            row_indices[ij] = temp;
        }
    }
}

std::vector<GPUOrderByPlan::SortColumn> GPUOrderByPlan::parseOrderBy(const Table &table) const
{
    std::vector<SortColumn> sort_cols;
    const auto &headers = table.getHeaders();

    for (const auto *order_desc : order_exprs_)
    {
        if (order_desc->type != hsql::kOrderAsc && order_desc->type != hsql::kOrderDesc)
            throw std::runtime_error("Unsupported ORDER BY type");

        const hsql::Expr *expr = order_desc->expr;
        if (!expr || expr->type != hsql::kExprColumnRef)
            throw std::runtime_error("Only column references are supported in ORDER BY");

        std::string col_name = expr->name;
        size_t col_idx = table.getColumnIndex(col_name);
        if (col_idx >= headers.size() || headers[col_idx] != col_name)
        {
            // Verify index matches header
            for (size_t i = 0; i < headers.size(); ++i)
            {
                if (headers[i] == col_name)
                {
                    col_idx = i;
                    break;
                }
            }
        }

        ColumnType col_type = table.getColumnType(col_name);
        sort_cols.push_back({col_idx, order_desc->type == hsql::kOrderAsc, col_type});
    }

    return sort_cols;
}

std::shared_ptr<Table> GPUOrderByPlan::execute()
{
    // Execute input plan
    std::shared_ptr<Table> table = input_->execute();

    if (!table || table->getData().empty())
        return table;

    // Parse ORDER BY
    std::vector<SortColumn> sort_cols = parseOrderBy(*table);
    if (sort_cols.empty())
        return table;

    // Get table dimensions
    const auto &data_map = table->getData();
    const auto &headers = table->getHeaders();
    size_t num_rows = data_map.begin()->second.size();
    size_t num_cols = headers.size();
    size_t pow2_size = 1;
    while (pow2_size < num_rows)
    {
        pow2_size *= 2;
    }

    // Prepare data arrays
    std::vector<int> int_data(pow2_size * num_cols, 0);
    std::vector<double> double_data(pow2_size * num_cols, 0.0);
    std::vector<char> string_data;
    std::vector<size_t> string_offsets(pow2_size * num_cols, 0); // Flat array for all columns
    size_t string_data_offset = 0;

    // Populate data in headers order
    for (size_t row_idx = 0; row_idx < num_rows; ++row_idx)
    {
        for (size_t col_idx = 0; col_idx < num_cols; ++col_idx)
        {
            const std::string &col_name = headers[col_idx];
            ColumnType col_type = table->getColumnType(col_name);
            const unionV &val = data_map.at(col_name)[row_idx];
            size_t flat_idx = row_idx * num_cols + col_idx;

            switch (col_type)
            {
            case ColumnType::INTEGER:
                int_data[flat_idx] = val.i;
                break;
            case ColumnType::DOUBLE:
                double_data[flat_idx] = val.d;
                break;
            case ColumnType::DATETIME:
                double_data[flat_idx] = val.d; // Handle DATETIME as DOUBLE
                break;
            case ColumnType::STRING:
            {
                std::string str = *val.s;
                string_offsets[flat_idx] = string_data_offset;
                for (char c : str)
                {
                    string_data.push_back(c);
                }
                string_data.push_back('\0');
                string_data_offset += str.length() + 1;
                break;
            }
            default:
                throw SemanticError("Unsupported column type in GPU ORDER BY");
            }
        }
    }

    // Pad string offsets for remaining rows
    for (size_t row_idx = num_rows; row_idx < pow2_size; ++row_idx)
    {
        for (size_t col_idx = 0; col_idx < num_cols; ++col_idx)
        {
            size_t flat_idx = row_idx * num_cols + col_idx;
            string_offsets[flat_idx] = string_data_offset;
        }
    }

    // Convert SortColumn to GPU-compatible struct
    std::vector<GPUSortColumn> gpu_sort_cols;
    for (const auto &col : sort_cols)
    {
        gpu_sort_cols.push_back({col.column_index, col.is_ascending, static_cast<int>(col.type)});
    }

    // Allocate GPU memory
    int *d_int_data = nullptr;
    double *d_double_data = nullptr;
    char *d_string_data = nullptr;
    size_t *d_string_offsets = nullptr;
    GPUSortColumn *d_sort_cols = nullptr;
    RowIndexValue *d_row_indices = nullptr;

    try
    {
        CUDA_CHECK(cudaMalloc(&d_int_data, int_data.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_double_data, double_data.size() * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_string_data, string_data.size() * sizeof(char)));
        CUDA_CHECK(cudaMalloc(&d_string_offsets, string_offsets.size() * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_sort_cols, gpu_sort_cols.size() * sizeof(GPUSortColumn)));
        CUDA_CHECK(cudaMalloc(&d_row_indices, pow2_size * sizeof(RowIndexValue)));

        CUDA_CHECK(cudaMemcpy(d_int_data, int_data.data(), int_data.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_double_data, double_data.data(), double_data.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_string_data, string_data.data(), string_data.size() * sizeof(char), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_string_offsets, string_offsets.data(), string_offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sort_cols, gpu_sort_cols.data(), gpu_sort_cols.size() * sizeof(GPUSortColumn), cudaMemcpyHostToDevice));

        // Initialize row indices on host
        std::vector<RowIndexValue> row_indices(pow2_size);
        for (size_t i = 0; i < pow2_size; ++i)
        {
            row_indices[i].row_index = i;
        }
        CUDA_CHECK(cudaMemcpy(d_row_indices, row_indices.data(), pow2_size * sizeof(RowIndexValue), cudaMemcpyHostToDevice));

        int threadsPerBlock = 256;
        int blocksPerGrid = (pow2_size + threadsPerBlock - 1) / threadsPerBlock;

        // Perform bitonic sort
        for (unsigned int k = 2; k <= pow2_size; k <<= 1)
        {
            for (unsigned int j = k >> 1; j > 0; j >>= 1)
            {
                bitonicSortStep<<<blocksPerGrid, threadsPerBlock>>>(
                    d_row_indices, pow2_size, j, k, d_sort_cols, gpu_sort_cols.size(), num_cols, num_rows,
                    d_int_data, d_double_data, d_string_data, d_string_offsets);
                CUDA_CHECK(cudaGetLastError());
            }
        }

        // Copy sorted indices back
        std::vector<RowIndexValue> sorted_row_indices(pow2_size);
        CUDA_CHECK(cudaMemcpy(sorted_row_indices.data(), d_row_indices, pow2_size * sizeof(RowIndexValue), cudaMemcpyDeviceToHost));

        // Create sorted table
        std::unordered_map<std::string, std::vector<unionV>> sorted_data_map;
        for (const auto &header : headers)
        {
            sorted_data_map[header].reserve(num_rows);
        }
        for (size_t i = 0; i < num_rows; ++i)
        {
            size_t orig_row_idx = sorted_row_indices[i].row_index;
            for (const auto &header : headers)
            {
                sorted_data_map[header].push_back(data_map.at(header)[orig_row_idx]);
            }
        }

        // Free GPU memory
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
    }
    catch (const std::exception &e)
    {
        // Clean up
        if (d_int_data)
            cudaFree(d_int_data);
        if (d_double_data)
            cudaFree(d_double_data);
        if (d_string_data)
            cudaFree(d_string_data);
        if (d_string_offsets)
            cudaFree(d_string_offsets);
        if (d_sort_cols)
            cudaFree(d_sort_cols);
        if (d_row_indices)
            cudaFree(d_row_indices);
        throw; // Propagate error
    }
}