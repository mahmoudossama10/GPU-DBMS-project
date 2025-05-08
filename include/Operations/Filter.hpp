#pragma once
#include <hsql/SQLParser.h>

#include "../DataHandling/Table.hpp"
#include "../QueryProcessing/PlanBuilder.hpp"

class FilterPlan : public ExecutionPlan
{
public:
    FilterPlan(std::unique_ptr<ExecutionPlan> input,
               const std::string condition)
        : input_(std::move(input)), string_condition(condition) {}

    std::shared_ptr<Table> execute() override;

private:
    std::unique_ptr<ExecutionPlan> input_;
    const std::string string_condition;
    const hsql::Expr *condition_;
};

class Filter
{
public:
    static std::shared_ptr<Table> apply(
        std::shared_ptr<Table> table,
        const hsql::Expr *condition);

private:
    // Evaluate the condition for a specific row index
    static bool evaluateCondition(
        const std::shared_ptr<Table> &table,
        size_t rowIndex,
        const hsql::Expr *condition);

    static bool handleOperator(
        const std::shared_ptr<Table> &table,
        size_t rowIndex,
        const hsql::Expr *expr);

    static bool handleNullCondition(
        const std::shared_ptr<Table> &table,
        size_t rowIndex,
        const hsql::Expr *expr);

    static bool matchLikePattern(const std::string &value, const std::string &pattern);

    // Type-specific comparison handlers
    static bool compareStrings(const std::string &lhs, const std::string &rhs, hsql::OperatorType op);
    static bool compareIntegers(unionV lhs, unionV rhs, hsql::OperatorType op);
    static bool compareDoubles(unionV lhs, unionV rhs, hsql::OperatorType op);
    static bool compareDateTimes(const dateTime &lhs, const dateTime &rhs, hsql::OperatorType op);

    static bool handleComparison(
        const std::shared_ptr<Table> &table,
        size_t rowIndex,
        const hsql::Expr *expr);

    // Get the appropriate value from an expression
    static unionV getExprValue(
        const std::shared_ptr<Table> &table,
        size_t rowIndex,
        const hsql::Expr *expr,
        ColumnType &outType);

    // Convert value to string for LIKE operations and debugging
    static std::string unionToString(const unionV &value, ColumnType type);
};