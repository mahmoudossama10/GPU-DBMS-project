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

    void evaluateTwoTableJoinCondition(
        const std::shared_ptr<Table> &leftTable,
        const std::shared_ptr<Table> &rightTable,
        hsql::Expr *condition, int direction);

    std::shared_ptr<Table> executeMultipleTableJoin(
        const std::vector<std::shared_ptr<Table>> &tables,
        const hsql::Expr *joinConditions);

    std::shared_ptr<Table> executeFilter(
        std::shared_ptr<Table> table,
        const hsql::Expr *where_clause);

    std::shared_ptr<Table> executeOrderBy(
        std::shared_ptr<Table> table,
        const std::vector<hsql::OrderDescription *> &order_exprs_);

    std::shared_ptr<Table> executeAggregate(
        std::shared_ptr<Table> table,
        const std::vector<hsql::Expr *> &select_list_);

    struct SortColumn
    {
        size_t column_index;
        bool is_ascending;
        ColumnType type;
    };

    struct AggregateOp
    {
        std::string function_name;
        std::string column_name;
        std::string alias;
        bool is_distinct;
        size_t column_index;
    };

    struct FilterCondition
    {
        size_t left_col_idx;
        size_t right_col_idx;
        bool is_literal;
        unionV literal_value;
        hsql::OperatorType op;
        ColumnType col_type;
    };

    std::shared_ptr<Table> output_join_table;
    int joinPlansCount = 0;

private:
    bool hasGPU_;

    int findColumnIndex(const Table &table, const char *columnName, const char *tableName);

    void processBatchesRecursive(
        const std::vector<std::shared_ptr<Table>> &tables,
        std::vector<std::vector<int>> &batchIndices,
        const hsql::Expr *joinConditions,
        std::vector<std::vector<unionV>> &resultData,
        int tableIndex);

    void processBatch(
        const std::vector<std::shared_ptr<Table>> &tables,
        const hsql::Expr *conditions, int direction);

    void evaluateConditionOnBatch(
        const std::vector<std::shared_ptr<Table>> &tables,
        const hsql::Expr *condition, int direction);

    std::vector<std::string> combineMultipleHeaders(
        const std::vector<std::shared_ptr<Table>> &tables);

    std::vector<std::vector<unionV>> mergeBatchResults(
        const std::vector<std::shared_ptr<Table>> &tables,
        const std::vector<std::vector<int>> &selectedIndices);

    std::vector<SortColumn> parseOrderBy(const Table &table, const std::vector<hsql::OrderDescription *> &order_exprs_);
    std::vector<AggregateOp> parseAggregates(const std::vector<hsql::Expr *> &select_list, const Table &table);
    std::shared_ptr<Table> aggregateTableGPU(const Table &table, const std::vector<AggregateOp> &aggregates);
    std::string unionValueToString(const unionV &value, ColumnType type);

    bool parseFilterConditions(
        const std::shared_ptr<Table> &table,
        const hsql::Expr *expr,
        std::vector<FilterCondition> &conditions);

    bool isComparisonOperator(hsql::OperatorType op);
};