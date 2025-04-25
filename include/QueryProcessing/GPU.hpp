#pragma once

#include <memory>
#include <vector>
#include <string>
#include "../DataHandling/Table.hpp"
#include "../DataHandling/StorageManager.hpp"
#include <hsql/sql/Expr.h>
#include <hsql/util/sqlhelper.h>
#include <hsql/SQLParser.h>

class GPUManager
{
public:
    GPUManager();
    ~GPUManager();

    // Join two tables with conditions on GPU
    // Returns a vector of booleans indicating which row combinations match

    std::shared_ptr<Table> executeJoin(std::shared_ptr<Table> leftTable,
                                       std::shared_ptr<Table> rightTable,
                                       const hsql::Expr *condition);

    std::pair<std::vector<std::pair<int, int>>, int>  gpuJoinTables(
        const Table &leftTable,
        const Table &rightTable,
        const hsql::Expr *conditions);

    // Filter a single table on GPU
    // Returns a vector of booleans indicating which rows match
    std::vector<uint8_t> gpuFilterTable(
        const Table &table,
        const hsql::Expr *conditions);

    // New filter functions
    std::shared_ptr<Table> applyFilter(const Table &table,
                                       const std::vector<uint8_t> &mask);

    std::vector<std::string> combineHeaders(const Table &left,
                                            const Table &right) const;

    std::vector<std::vector<std::string>> mergeJoinResults(
        const Table &left,
        const Table &right,
        const std::vector<std::pair<int, int>>& joinPairs) const;

    std::vector<std::vector<std::string>> mergeFilterResults(
        const Table &table,
        const std::vector<uint8_t> &mask) const;

     hsql::Expr* simplifyCondition(hsql::Expr* expr);
    std::vector<std::string> extractColumnReferences(hsql::Expr* expr);
    int getParentOperationType(hsql::Expr* expr);
    bool canExecuteCondition(std::shared_ptr<Table> table, hsql::Expr* condition);
    hsql::Expr* filterJoinCondition(std::shared_ptr<Table> leftTable,
        std::shared_ptr<Table> rightTable,
        hsql::Expr* condition);

    // Check if GPU operations are available
    bool isGPUAvailable() const;

private:
    // Prepare expression for GPU execution
    void prepareExpression(const hsql::Expr *expr);

    // Find column indices for comparison
    int findColumnIndex(const Table &table, const char *columnName, const char *tableName = nullptr);

    // Handle different expression types
    std::vector<uint8_t> processComparisonExpr(
        const Table &leftTable,
        const Table &rightTable,
        const hsql::Expr *expr);

    std::vector<uint8_t> processBinaryExpr(
        const Table &leftTable,
        const Table &rightTable,
        const hsql::Expr *expr);

    bool hasGPU_;
};
