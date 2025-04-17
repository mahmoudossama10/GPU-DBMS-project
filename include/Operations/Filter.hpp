#pragma once
#include <hsql/SQLParser.h>

#include "../DataHandling/Table.hpp"
#include "../QueryProcessing/PlanBuilder.hpp"

class FilterPlan : public ExecutionPlan
{
public:
    FilterPlan(std::unique_ptr<ExecutionPlan> input,
               const hsql::Expr *condition)
        : input_(std::move(input)), condition_(condition) {}

    std::shared_ptr<Table> execute() override;

private:
    std::unique_ptr<ExecutionPlan> input_;
    const hsql::Expr *condition_;
};

class Filter
{
public:
    static std::shared_ptr<Table> apply(
        std::shared_ptr<Table> table,
        const hsql::Expr *condition);

private:
    static bool evaluateCondition(
        const std::vector<std::string> &row,
        const hsql::Expr *condition,
        const std::vector<std::string> &headers,
        const std::unordered_map<std::string, size_t> &columnIndexMap);

    static bool handleOperator(
        const std::vector<std::string> &row,
        const hsql::Expr *expr,
        const std::vector<std::string> &headers,
        const std::unordered_map<std::string, size_t> &columnIndexMap);

    static bool handleNullCondition(
        const std::vector<std::string> &row,
        const hsql::Expr *expr,
        const std::unordered_map<std::string, size_t> &columnIndexMap);

    static bool matchLikePattern(const std::string &value, const std::string &pattern);
    static bool compareStrings(const std::string &lhs, const std::string &rhs, hsql::OperatorType op);
    static bool compareNumerics(double lhs, double rhs, hsql::OperatorType op);
    static bool handleComparison(
        const std::vector<std::string> &row,
        const hsql::Expr *expr,
        const std::vector<std::string> &headers,
        const std::unordered_map<std::string, size_t> &columnIndexMap);

    static bool compareValues(
        const std::string &lhs,
        const std::string &rhs,
        hsql::OperatorType op);

    static bool handleLikeOperator(const std::string &value, const std::string &pattern);
};