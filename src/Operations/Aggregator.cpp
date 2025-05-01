// #include "../../include/Operations/Aggregator.hpp"
// #include "../../include/Utilities/ErrorHandling.hpp"
// #include "../../include/Utilities/StringUtils.hpp"
// #include <algorithm>
// #include <stdexcept>
// #include <cmath> // For NaN
// #include "../../include/Operations/GPUAggregator.hpp"
// #include <iostream>
// #include <unordered_map>
// using namespace StringUtils; // For case-insensitive comparison

// // Constructor
// AggregatorPlan::AggregatorPlan(std::unique_ptr<ExecutionPlan> input,
//                                const std::vector<hsql::Expr *> &aggregate_exprs)
//     : input_(std::move(input)), aggregate_exprs_(aggregate_exprs) {}

// // Main execution logic
// std::shared_ptr<Table> AggregatorPlan::execute()
// {
//     auto input_table = input_->execute();
//     auto aggregates = parseAggregateExpressions();

//     std::vector<std::string> headers;
//     std::vector<std::string> result_row;

//     // ------ NEW: map from column name to computed IntAggregates (GPU results) ------
//     std::unordered_map<std::string, GPUAggregator::IntAggregates> int_col_aggregates;

//     // ------ NEW: track which columns we've attempted batched GPU on ------
//     std::unordered_map<std::string, bool> did_gpu_aggregate;

//     for (const auto &agg : aggregates)
//     {

//         double value = NAN;

//         int count = 0;

//         bool isIntCol = false;
//         bool useGPU = true;

//         // -------- Handle COUNT(*) fast-path -----------

//         if (agg.type == AggregateInfo::AggType::COUNT && agg.column.empty())
//         {

//             result_row.push_back(std::to_string(input_table->getSize()));

//             headers.push_back(!agg.alias.empty() ? agg.alias : "COUNT(*)");

//             continue;
//         }

//         // -------- Batched int-aggregate logic ----------

//         if (!agg.column.empty())
//         {

//             // Only extract/calculate for a column if not already done!

//             if (did_gpu_aggregate.find(agg.column) == did_gpu_aggregate.end())
//             {
//                 try
//                 {
//                     // ----------- NEW: Use the columnar cache! -----------
//                     const auto &col_data = input_table->getIntColumn(agg.column);
//                     isIntCol = true;
//                     if (useGPU)
//                     {
//                         int_col_aggregates[agg.column] = GPUAggregator::multiAggregateInt(col_data);
//                         did_gpu_aggregate[agg.column] = true;
//                     }
//                     else
//                     {
//                         did_gpu_aggregate[agg.column] = false;
//                     }
//                 }
//                 catch (...)
//                 {
//                     // If getIntColumn fails, treat as non-int
//                     isIntCol = false;
//                     did_gpu_aggregate[agg.column] = false;
//                 }
//             }
//             else
//             {
//                 isIntCol = did_gpu_aggregate[agg.column];
//             }
//         }

//         // --------- Aggregate result assignment ---------

//         // Fill result_row, either using GPU-batched result or CPU fallback

//         if (useGPU && !agg.column.empty() && isIntCol && int_col_aggregates.find(agg.column) != int_col_aggregates.end())
//         {

//             // Use the batch results

//             const auto &aggs = int_col_aggregates[agg.column];

//             switch (agg.type)

//             {

//             case AggregateInfo::AggType::COUNT:

//                 result_row.push_back(std::to_string(aggs.count));

//                 break;

//             case AggregateInfo::AggType::SUM:

//                 result_row.push_back(std::to_string(aggs.sum));

//                 break;

//             case AggregateInfo::AggType::AVG:

//                 result_row.push_back(std::to_string(aggs.avg));

//                 break;

//             case AggregateInfo::AggType::MAX:

//                 result_row.push_back(std::to_string(aggs.max));

//                 break;

//             case AggregateInfo::AggType::MIN:

//                 result_row.push_back(std::to_string(aggs.min));

//                 break;
//             }
//         }
//         else
//         {

//             // ----------- CPU fallback as before -----------

//             switch (agg.type)

//             {

//             case AggregateInfo::AggType::COUNT:

//                 count = computeCount(*input_table, agg.column);

//                 result_row.push_back(std::to_string(count));

//                 break;

//             case AggregateInfo::AggType::SUM:

//                 value = computeSum(*input_table, agg.column);

//                 result_row.push_back(std::to_string(value));

//                 break;

//             case AggregateInfo::AggType::AVG:

//                 value = computeAvg(*input_table, agg.column);

//                 result_row.push_back(std::to_string(value));

//                 break;

//             case AggregateInfo::AggType::MAX:

//                 value = computeMax(*input_table, agg.column);

//                 result_row.push_back(std::to_string(value));

//                 break;

//             case AggregateInfo::AggType::MIN:

//                 value = computeMin(*input_table, agg.column);

//                 result_row.push_back(std::to_string(value));

//                 break;
//             }
//         }

//         // ----------- Always build safe/correct headers -----------

//         headers.push_back(!agg.alias.empty() ? agg.alias

//                                              : (agg.column.empty() ? aggToString(agg.type) + "(*)"

//                                                                    : aggToString(agg.type) + "(" + agg.column + ")"));
//     }

//     // Create output table
//     auto output_table = std::make_shared<Table>(
//         input_table->getName() + "_aggregated",
//         headers,
//         std::vector<std::vector<std::string>>{result_row});

//     return output_table;
// }

// // --- Helper Functions ---

// // Parse HSQL expressions into aggregate operations

// std::vector<AggregatorPlan::AggregateInfo> AggregatorPlan::parseAggregateExpressions() const
// {
//     std::vector<AggregateInfo> aggregates;

//     for (const auto *expr : aggregate_exprs_)
//     {
//         if (expr->type != hsql::kExprFunctionRef)
//         {
//             throw SemanticError("Non-aggregate expression in aggregate context");
//         }

//         AggregateInfo info;
//         info.alias = expr->alias ? expr->alias : "";
//         std::string func_name = toLower(expr->name);

//         // Map function name to type
//         if (func_name == "count")
//             info.type = AggregateInfo::AggType::COUNT;
//         else if (func_name == "sum")
//             info.type = AggregateInfo::AggType::SUM;
//         else if (func_name == "avg")
//             info.type = AggregateInfo::AggType::AVG;
//         else if (func_name == "max")
//             info.type = AggregateInfo::AggType::MAX;
//         else if (func_name == "min")
//             info.type = AggregateInfo::AggType::MIN;
//         else
//             throw SemanticError("Unsupported aggregate function: " + func_name);

//         // Handle COUNT(*) vs COUNT(col)
//         if (expr->exprList && !expr->exprList->empty())
//         {
//             const auto *arg = expr->exprList->at(0);
//             if (arg->type == hsql::kExprStar)
//             {
//                 info.column = ""; // COUNT(*)
//             }
//             else
//             {
//                 info.column = getColumnName(arg);
//             }
//         }

//         aggregates.push_back(info);
//     }

//     return aggregates;
// }

// std::string AggregatorPlan::getColumnName(const hsql::Expr *expr) const
// {
//     if (expr->type != hsql::kExprColumnRef)
//     {
//         throw SemanticError("Complex expressions in aggregates not supported");
//     }

//     // Handle aliased columns (table.column)
//     if (expr->table != nullptr && expr->table[0] != '\0')
//     {
//         return std::string(expr->table) + "." + expr->name;
//     }
//     return expr->name;
// }

// std::string AggregatorPlan::aggToString(AggregateInfo::AggType type) const
// {
//     switch (type)
//     {
//     case AggregateInfo::AggType::COUNT:
//         return "COUNT";
//     case AggregateInfo::AggType::SUM:
//         return "SUM";
//     case AggregateInfo::AggType::AVG:
//         return "AVG";
//     case AggregateInfo::AggType::MAX:
//         return "MAX";
//     case AggregateInfo::AggType::MIN:
//         return "MIN";
//     default:
//         throw std::runtime_error("Unknown aggregate type");
//     }
// }

// // --- Aggregate Computations ---

// double AggregatorPlan::computeSum(const Table &table, const std::string &column) const
// {
//     double sum = 0.0;

//     // Get the column data directly from the column-major structure
//     const auto &columnData = table.getData();

//     // Check if column exists
//     if (columnData.find(column) == columnData.end())
//     {
//         throw std::runtime_error("Column not found: " + column);
//     }

//     // Access the specific column vector
//     const auto &columnValues = columnData.at(column);

//     // Sum each value in the column
//     for (const auto &value : columnValues)
//     {
//         sum += numericColumnSafeGet(table, value);
//     }

//     return sum;
// }

// double AggregatorPlan::computeAvg(const Table &table, const std::string &column) const
// {
//     double sum = computeSum(table, column);
//     int count = computeCount(table, column);
//     return (count == 0) ? NAN : sum / count;
// }

// double AggregatorPlan::computeMax(const Table &table, const std::string &column) const
// {
//     double max_val = -INFINITY;

//     // Get the column data directly from the column-major structure
//     const auto &columnData = table.getData();

//     // Check if column exists
//     if (columnData.find(column) == columnData.end())
//     {
//         throw std::runtime_error("Column not found: " + column);
//     }

//     // Access the specific column vector
//     const auto &columnValues = columnData.at(column);

//     // Find maximum value in the column
//     for (const auto &value : columnValues)
//     {
//         double val = numericColumnSafeGet(table, value);
//         if (val > max_val)
//             max_val = val;
//     }

//     return max_val;
// }

// double AggregatorPlan::computeMin(const Table &table, const std::string &column) const
// {
//     double min_val = INFINITY;

//     // Get the column data directly from the column-major structure
//     const auto &columnData = table.getData();

//     // Check if column exists
//     if (columnData.find(column) == columnData.end())
//     {
//         throw std::runtime_error("Column not found: " + column);
//     }

//     // Access the specific column vector
//     const auto &columnValues = columnData.at(column);

//     // Find minimum value in the column
//     for (const auto &value : columnValues)
//     {
//         double val = numericColumnSafeGet(table, value);
//         if (val < min_val)
//             min_val = val;
//     }

//     return min_val;
// }

// int AggregatorPlan::computeCount(const Table &table, const std::string &column) const
// {
//     return column.empty() ? table.getSize() : table.getSize(); // COUNT(*) vs COUNT(col)
// }

// // Safe numeric conversion with error handling
// double AggregatorPlan::numericColumnSafeGet(const Table &table, const std::string &value) const
// {
//     try
//     {
//         return std::stod(value);
//     }
//     catch (...)
//     {
//         throw SemanticError("Non-numeric value in numeric aggregate column: " + value);
//     }
// }