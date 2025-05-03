#pragma once
#include "../DataHandling/Table.hpp"
#include "../QueryProcessing/PlanBuilder.hpp"
#include <cuda_runtime.h>
#include "Utilities/ErrorHandling.hpp"
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                  \
    do                                                                    \
    {                                                                     \
        cudaError_t error = call;                                         \
        if (error != cudaSuccess)                                         \
        {                                                                 \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__; \
            std::cerr << ": " << cudaGetErrorString(error) << std::endl;  \
            throw std::runtime_error("CUDA error");                       \
        }                                                                 \
    } while (0)

class GPUOrderByPlan : public ExecutionPlan
{
public:
    GPUOrderByPlan(std::unique_ptr<ExecutionPlan> input,
                   const std::vector<hsql::OrderDescription *> &order_exprs);
    std::shared_ptr<Table> execute() override;

private:
    std::unique_ptr<ExecutionPlan> input_;
    std::vector<hsql::OrderDescription *> order_exprs_;

    struct SortColumn
    {
        size_t column_index;
        bool is_ascending;
        ColumnType type;
    };

    std::vector<SortColumn> parseOrderBy(const Table &table) const;
};