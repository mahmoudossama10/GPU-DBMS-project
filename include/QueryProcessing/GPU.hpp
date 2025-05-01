#pragma once

#include <memory>
#include <vector>
#include <string>
#include "../DataHandling/Table.hpp"
#include "../DataHandling/StorageManager.hpp"
#include <hsql/sql/Expr.h>
#include <hsql/util/sqlhelper.h>
#include <hsql/SQLParser.h>


class GPUManager {
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
        const std::vector<std::shared_ptr<Table>>& tables,
        const hsql::Expr* joinConditions);

private:
    // Flag indicating if GPU is available
    bool hasGPU_;

    // Helper method to find column index
    int findColumnIndex(const Table& table, const char* columnName, const char* tableName);

    // Process batches recursively
    void processBatchesRecursive(
        const std::vector<std::shared_ptr<Table>>& tables,
        std::vector<std::vector<int>>& batchIndices,
        const hsql::Expr* joinConditions,
        std::vector<std::vector<unionV>>& resultData,
        int tableIndex);

    // Process a specific batch
    std::vector<uint8_t> processBatch(
        const std::vector<std::shared_ptr<Table>>& tables,
        const std::vector<std::vector<int>>& batchIndices,
        const hsql::Expr* conditions);

    // Evaluate condition on batch
    std::vector<uint8_t> evaluateConditionOnBatch(
        const std::vector<std::shared_ptr<Table>>& tables,
        const std::vector<std::vector<int>>& batchIndices,
        const hsql::Expr* condition);

    // Combine headers from multiple tables
    std::vector<std::string> combineMultipleHeaders(
        const std::vector<std::shared_ptr<Table>>& tables);

    // Merge rows based on selected indices
    std::vector<std::vector<unionV>> mergeBatchResults(
        const std::vector<std::shared_ptr<Table>>& tables,
        const std::vector<std::vector<int>>& selectedIndices);
};