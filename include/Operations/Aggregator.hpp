#pragma once
#include "../DataHandling/Table.hpp"
#include "../QueryProcessing/PlanBuilder.hpp"
#include <hsql/SQLParser.h>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

class AggregatorPlan : public ExecutionPlan {
public:
    AggregatorPlan(std::unique_ptr<ExecutionPlan> input, 
                   const std::vector<hsql::Expr*>& aggregate_exprs);
    
    std::shared_ptr<Table> execute() override;

private:
    std::unique_ptr<ExecutionPlan> input_;
    std::vector<hsql::Expr*> aggregate_exprs_;

    // Helper structures
    struct AggregateInfo {
        enum class AggType { COUNT, SUM, AVG, MAX, MIN };
        AggType type;
        std::string column;  // Column to aggregate (empty for COUNT(*))
        std::string alias;
    };
    std::string aggToString(AggregateInfo::AggType type) const;

    // Key utilities
    std::vector<AggregateInfo> parseAggregateExpressions() const;
    std::string getColumnName(const hsql::Expr* expr) const;
    double numericColumnSafeGet(const Table& table, const std::string& column) const;
    
    // Computation functions
    double computeSum(const Table& table, const std::string& column) const;
    double computeAvg(const Table& table, const std::string& column) const;
    double computeMax(const Table& table, const std::string& column) const;
    double computeMin(const Table& table, const std::string& column) const;
    int computeCount(const Table& table, const std::string& column) const;
};