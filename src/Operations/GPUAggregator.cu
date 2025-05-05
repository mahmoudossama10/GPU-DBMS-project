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

// Tunable parameters based on GPU architecture (not used in CPU fallback)
#define THREADS_PER_BLOCK 512
#define COARSENING_FACTOR 8

// Structure to hold aggregation results (not used in CPU fallback)
struct AggResult {
    double sum_val;
    double min_val;
    double max_val;
    int64_t count_val;
};

// Dummy kernel (not used in current implementation due to CPU fallback)
__global__ void combinedAggKernel(const double* data, size_t num_rows, AggResult* block_results, size_t coarsening_factor) {
    // Placeholder - not used since we're falling back to CPU
}

// Dummy kernel (not used in current implementation due to CPU fallback)
__global__ void finalReductionKernel(const AggResult* block_results, size_t num_blocks, AggResult* final_result) {
    // Placeholder - not used since we're falling back to CPU
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

    // Force CPU path for correctness until GPU implementation is fixed
    for (const auto& op : aggregates) {
        result_headers.push_back(op.alias);
        ColumnType col_type = op.column_name.empty() ? ColumnType::INTEGER : table.getColumnType(op.column_name);
        result_types[op.alias] = (op.function_name == "count") ? ColumnType::INTEGER : col_type;

        std::vector<unionV> result_col(1);
        if (op.function_name == "count") {
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
                result_col[0].i = num_rows;
            }
        } else if (col_type == ColumnType::STRING) {
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
            if (op.is_distinct) {
                throw SemanticError("DISTINCT on numeric types not supported in GPU fallback yet");
            }
            if (op.function_name == "sum" || op.function_name == "avg") {
                double total = 0.0;
                for (size_t i = 0; i < num_rows; ++i) {
                    if (col_type == ColumnType::INTEGER) {
                        total += static_cast<double>(table.getInteger(op.column_name, i));
                    } else if (col_type == ColumnType::DOUBLE) {
                        total += table.getDouble(op.column_name, i);
                    }
                }
                if (op.function_name == "avg" && num_rows > 0) {
                    total /= static_cast<double>(num_rows);
                }
                if (col_type == ColumnType::INTEGER && op.function_name != "avg") {
                    result_col[0].i = static_cast<int64_t>(total);
                } else {
                    result_col[0].d = total;
                }
            } else if (op.function_name == "min" || op.function_name == "max") {
                double extreme = (op.function_name == "min") ? 1.0e308 : -1.0e308;
                for (size_t i = 0; i < num_rows; ++i) {
                    double val = (col_type == ColumnType::INTEGER) ? static_cast<double>(table.getInteger(op.column_name, i)) : table.getDouble(op.column_name, i);
                    if (op.function_name == "min" ? val < extreme : val > extreme) {
                        extreme = val;
                    }
                }
                if (col_type == ColumnType::INTEGER) {
                    result_col[0].i = static_cast<int64_t>(extreme);
                } else {
                    result_col[0].d = extreme;
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