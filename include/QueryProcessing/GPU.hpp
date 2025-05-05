#pragma once

#include <memory>
#include <vector>
#include <string>
#include "../DataHandling/Table.hpp"
#include "../DataHandling/StorageManager.hpp"
#include <hsql/sql/Expr.h>
#include <hsql/util/sqlhelper.h>
#include <hsql/SQLParser.h>

#include "Utilities/ErrorHandling.hpp"
#include <unordered_map>

// Define a structure that can be passed to the GPU
struct GPUSortColumn
{
    size_t column_index;
    bool is_ascending;
    int type; // Using int instead of enum for simplicity on GPU
};

// Structure for row indices
struct RowIndexValue
{
    size_t row_index; // Original row index - needed for stable sort
};

class GPUManager
{
public:
    // Constant for batch size
    static const int BATCH_SIZE;

    // Constructor and destructor
    GPUManager();
    ~GPUManager();

    // Check if GPU is available
    bool isGPUAvailable() const;

    // Execute join with batch processing
    std::shared_ptr<Table> executeBatchedJoin(
        const std::vector<std::shared_ptr<Table>> &tables,
        const hsql::Expr *joinConditions);

    std::shared_ptr<Table> executeTwoTableJoinWithBinarySearch(
        const std::shared_ptr<Table> &leftTable,
        const std::shared_ptr<Table> &rightTable,
        hsql::Expr *joinCondition);

    std::vector<int64_t> evaluateTwoTableJoinCondition(
        const std::shared_ptr<Table> &leftTable,
        const std::shared_ptr<Table> &rightTable,
        hsql::Expr *condition);

    std::shared_ptr<Table> executeMultipleTableJoin(
        const std::vector<std::shared_ptr<Table>> &tables,
        const hsql::Expr *joinConditions);

    std::shared_ptr<Table> executeOrderBy(
        std::shared_ptr<Table> table,
        const std::vector<hsql::OrderDescription *> &order_exprs_);
    struct SortColumn
    {
        size_t column_index;
        bool is_ascending;
        ColumnType type;
    };

private:
    // Flag indicating if GPU is available
    bool hasGPU_;

    // Helper method to find column index
    int findColumnIndex(const Table &table, const char *columnName, const char *tableName);

    // Process batches recursively
    void processBatchesRecursive(
        const std::vector<std::shared_ptr<Table>> &tables,
        std::vector<std::vector<int>> &batchIndices,
        const hsql::Expr *joinConditions,
        std::vector<std::vector<unionV>> &resultData,
        int tableIndex);

    // Process a specific batch
    std::vector<int64_t> processBatch(
        const std::vector<std::shared_ptr<Table>> &tables,
        const hsql::Expr *conditions);

    // Evaluate condition on batch
    std::vector<int64_t> evaluateConditionOnBatch(
        const std::vector<std::shared_ptr<Table>> &tables,
        const hsql::Expr *condition);

    // Combine headers from multiple tables
    std::vector<std::string> combineMultipleHeaders(
        const std::vector<std::shared_ptr<Table>> &tables);

    // Merge rows based on selected indices
    std::vector<std::vector<unionV>> mergeBatchResults(
        const std::vector<std::shared_ptr<Table>> &tables,
        const std::vector<std::vector<int>> &selectedIndices);

    std::vector<SortColumn> parseOrderBy(const Table &table, const std::vector<hsql::OrderDescription *> &order_exprs_);
};