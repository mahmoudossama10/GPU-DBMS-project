// #pragma once
// #include "../DataHandling/Table.hpp"
// #include "../QueryProcessing/PlanBuilder.hpp"
// #include <cuda_runtime.h>
// #include "Utilities/ErrorHandling.hpp"
// #include <memory>
// #include <vector>
// #include <string>
// #include <unordered_map>
// #include <algorithm>
// #include <iostream>

// // CUDA error checking macro
// #define CUDA_CHECK(call)                                                  \
//     do                                                                    \
//     {                                                                     \
//         cudaError_t error = call;                                         \
//         if (error != cudaSuccess)                                         \
//         {                                                                 \
//             std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__; \
//             std::cerr << ": " << cudaGetErrorString(error) << std::endl;  \
//             throw std::runtime_error("CUDA error");                       \
//         }                                                                 \
//     } while (0)

// // Forward declarations for CUDA kernel functions
// struct GPUSortColumn;
// struct RowIndexValue;

// __global__ void prepareRowIndices(const int *int_data, const double *double_data,
//                                   const char *string_data, size_t *string_offsets,
//                                   RowIndexValue *row_indices, size_t num_rows, size_t num_cols,
//                                   GPUSortColumn *sort_columns, int num_sort_columns,
//                                   int primary_sort_column_index);

// __device__ bool compareRowIndices(const RowIndexValue &a, const RowIndexValue &b,
//                                  const GPUSortColumn *sort_columns, int num_sort_cols, size_t num_cols, size_t original_num_rows,
//                                  const int *int_data, const double *double_data,
//                                  const char *string_data, size_t *string_offsets);

// __global__ void bitonicSortStep(RowIndexValue *row_indices, size_t n,
//                                unsigned int j, unsigned int k,
//                                const GPUSortColumn *sort_columns, int num_sort_cols, size_t num_cols, size_t original_num_rows,
//                                const int *int_data, const double *double_data,
//                                const char *string_data, size_t *string_offsets);

// class GPUOrderByPlan : public ExecutionPlan
// {
// public:
//     GPUOrderByPlan(std::unique_ptr<ExecutionPlan> input,
//                    const std::vector<hsql::OrderDescription *> &order_exprs);
//     std::shared_ptr<Table> execute() override;

// private:
//     std::unique_ptr<ExecutionPlan> input_;
//     std::vector<hsql::OrderDescription *> order_exprs_;

//     struct SortColumn
//     {
//         size_t column_index;
//         bool is_ascending;
//         ColumnType type;
//     };

//     std::vector<SortColumn> parseOrderBy(const Table &table) const;
// };