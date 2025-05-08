#pragma once
#include <hsql/SQLParser.h>

#include "../DataHandling/Table.hpp"
#include "../QueryProcessing/PlanBuilder.hpp"
#include <vector>

class JoinPlan : public ExecutionPlan
{
public:
    JoinPlan(std::vector<std::unique_ptr<ExecutionPlan>> inputs,
             std::string where)
        : inputs_(std::move(inputs)), whereString(where) {}

    std::shared_ptr<Table> execute() override;

private:
    std::vector<std::unique_ptr<ExecutionPlan>> inputs_;
    const hsql::Expr *join_condition_;
    std::string whereString;
};

class Join
{
public:
    static std::shared_ptr<Table> apply(
        const std::vector<std::shared_ptr<Table>> &tables,
        const hsql::Expr *condition);

private:
    // Helper function to evaluate join condition for a combination of rows
    static bool evaluateJoinCondition(
        const std::vector<std::shared_ptr<Table>> &tables,
        const std::vector<size_t> &rowIndices,
        const hsql::Expr *condition);

    // Helper function to handle logical operators (AND, OR, NOT)
    static bool handleLogicalOperator(
        const std::vector<std::shared_ptr<Table>> &tables,
        const std::vector<size_t> &rowIndices,
        const hsql::Expr *expr);

    // Helper function to handle comparison operators
    static bool handleComparison(
        const std::vector<std::shared_ptr<Table>> &tables,
        const std::vector<size_t> &rowIndices,
        const hsql::Expr *expr);

    // Helper function to get a value from a specific row in a table
    static unionV getExprValue(
        const std::vector<std::shared_ptr<Table>> &tables,
        const std::vector<size_t> &rowIndices,
        const hsql::Expr *expr,
        ColumnType &outType,
        int &tableIndex);

    // Type-specific comparison helpers
    static bool compareValues(
        const unionV &lhs, ColumnType lhsType,
        const unionV &rhs, ColumnType rhsType,
        hsql::OperatorType op);

    static bool compareStrings(const std::string &lhs, const std::string &rhs, hsql::OperatorType op);
    static bool compareIntegers(int64_t lhs, int64_t rhs, hsql::OperatorType op);
    static bool compareDoubles(double lhs, double rhs, hsql::OperatorType op);
    static bool compareDateTimes(const dateTime &lhs, const dateTime &rhs, hsql::OperatorType op);

    // Helper function to combine headers from multiple tables
    static std::vector<std::string> combineHeaders(const std::vector<std::shared_ptr<Table>> &tables);

    // Helper function to find the table and column for a column reference
    static std::pair<int, std::string> findTableAndColumn(
        const std::vector<std::shared_ptr<Table>> &tables,
        const char *columnName,
        const char *tableName = nullptr);
};