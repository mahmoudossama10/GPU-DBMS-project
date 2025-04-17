#include "../../include/Operations/Aggregator.hpp"
#include "../../include/Utilities/ErrorHandling.hpp"
#include "../../include/Utilities/StringUtils.hpp"
#include <algorithm>
#include <stdexcept>
#include <cmath> // For NaN

using namespace StringUtils; // For case-insensitive comparison

// Constructor
AggregatorPlan::AggregatorPlan(std::unique_ptr<ExecutionPlan> input,
                               const std::vector<hsql::Expr*>& aggregate_exprs)
    : input_(std::move(input)), aggregate_exprs_(aggregate_exprs) {}

// Main execution logic
std::shared_ptr<Table> AggregatorPlan::execute() {
    auto input_table = input_->execute();
    auto aggregates = parseAggregateExpressions();
    
    std::vector<std::string> headers;
    std::vector<std::string> result_row;
    
    for (const auto& agg : aggregates) {
        // Compute each aggregate
        double value = NAN;
        int count = 0;
        
        switch (agg.type) {
            case AggregateInfo::AggType::COUNT:
                count = computeCount(*input_table, agg.column);
                result_row.push_back(std::to_string(count));
                break;
            case AggregateInfo::AggType::SUM:
                value = computeSum(*input_table, agg.column);
                result_row.push_back(std::to_string(value));
                break;
            case AggregateInfo::AggType::AVG:
                value = computeAvg(*input_table, agg.column);
                result_row.push_back(std::to_string(value));
                break;
            case AggregateInfo::AggType::MAX:
                value = computeMax(*input_table, agg.column);
                result_row.push_back(std::to_string(value));
                break;
            case AggregateInfo::AggType::MIN:
                value = computeMin(*input_table, agg.column);
                result_row.push_back(std::to_string(value));
                break;
        }
        
        // Build header with alias or generated name
        headers.push_back(!agg.alias.empty() ? agg.alias 
            : (agg.column.empty() ? aggToString(agg.type) + "(*)"
                                   : aggToString(agg.type) + "(" + agg.column + ")"));
    }

    // Create output table
    auto output_table = std::make_shared<Table>(
        input_table->getName() + "_aggregated",
        headers,
        std::vector<std::vector<std::string>>{result_row}
    );
    
    return output_table;
}

// --- Helper Functions ---

// Parse HSQL expressions into aggregate operations
std::vector<AggregatorPlan::AggregateInfo> AggregatorPlan::parseAggregateExpressions() const {
    std::vector<AggregateInfo> aggregates;
    
    for (const auto* expr : aggregate_exprs_) {
        if (expr->type != hsql::kExprFunctionRef) {
            throw SemanticError("Non-aggregate expression in aggregate context");
        }
        
        AggregateInfo info;
        info.alias = expr->alias ? expr->alias : "";
        std::string func_name = toLower(expr->name);
        
        // Map function name to type
        if (func_name == "count") info.type = AggregateInfo::AggType::COUNT;
        else if (func_name == "sum") info.type = AggregateInfo::AggType::SUM;
        else if (func_name == "avg") info.type = AggregateInfo::AggType::AVG;
        else if (func_name == "max") info.type = AggregateInfo::AggType::MAX;
        else if (func_name == "min") info.type = AggregateInfo::AggType::MIN;
        else throw SemanticError("Unsupported aggregate function: " + func_name);
        
        // Handle COUNT(*) vs COUNT(col)
        if (expr->exprList && !expr->exprList->empty()) {
            const auto* arg = expr->exprList->at(0);
            if (arg->type == hsql::kExprStar) {
                info.column = ""; // COUNT(*)
            } else {
                info.column = getColumnName(arg);
            }
        }
        
        aggregates.push_back(info);
    }
    
    return aggregates;
}

std::string AggregatorPlan::aggToString(AggregateInfo::AggType type) const {
    switch (type) {
        case AggregateInfo::AggType::COUNT: return "COUNT";
        case AggregateInfo::AggType::SUM:   return "SUM";
        case AggregateInfo::AggType::AVG:   return "AVG";
        case AggregateInfo::AggType::MAX:   return "MAX";
        case AggregateInfo::AggType::MIN:   return "MIN";
        default: throw std::runtime_error("Unknown aggregate type");
    }
}

// Get column name from expression (supports simple column refs)
std::string AggregatorPlan::getColumnName(const hsql::Expr* expr) const {
    if (expr->type != hsql::kExprColumnRef) {
        throw SemanticError("Complex expressions in aggregates not supported");
    }
    return expr->name;
}

// --- Aggregate Computations ---

double AggregatorPlan::computeSum(const Table& table, const std::string& column) const {
    double sum = 0.0;
    size_t col_idx = table.getColumnIndex(column);
    
    for (const auto& row : table.getData()) {
        sum += numericColumnSafeGet(table, row[col_idx]);
    }
    return sum;
}

double AggregatorPlan::computeAvg(const Table& table, const std::string& column) const {
    double sum = computeSum(table, column);
    int count = computeCount(table, column);
    return (count == 0) ? NAN : sum / count;
}

double AggregatorPlan::computeMax(const Table& table, const std::string& column) const {
    size_t col_idx = table.getColumnIndex(column);
    double max_val = -INFINITY;
    
    for (const auto& row : table.getData()) {
        double val = numericColumnSafeGet(table, row[col_idx]);
        if (val > max_val) max_val = val;
    }
    return max_val;
}

double AggregatorPlan::computeMin(const Table& table, const std::string& column) const {
    size_t col_idx = table.getColumnIndex(column);
    double min_val = INFINITY;
    
    for (const auto& row : table.getData()) {
        double val = numericColumnSafeGet(table, row[col_idx]);
        if (val < min_val) min_val = val;
    }
    return min_val;
}

int AggregatorPlan::computeCount(const Table& table, const std::string& column) const {
    return column.empty() ? table.getSize() : table.getSize(); // COUNT(*) vs COUNT(col)
}

// Safe numeric conversion with error handling
double AggregatorPlan::numericColumnSafeGet(const Table& table, const std::string& value) const {
    try {
        return std::stod(value);
    } catch (...) {
        throw SemanticError("Non-numeric value in numeric aggregate column: " + value);
    }
}