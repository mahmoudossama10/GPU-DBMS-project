#pragma once
#include <memory>
#include <hsql/SQLParser.h>
#include "../DataHandling/Table.hpp"
#include "../DataHandling/StorageManager.hpp"
#include "../QueryProcessing/PlanBuilder.hpp"

class QueryExecutor
{
public:
    explicit QueryExecutor(std::shared_ptr<StorageManager> storage);

    std::shared_ptr<Table> execute(const std::string &query);
    std::shared_ptr<Table> execute(const hsql::SelectStatement *stmt, const std::string &query);
    void mergeResults(std::shared_ptr<Table> table1,
                      std::shared_ptr<Table> table2);
    void cleanupBatchTables();
    std::string replaceTableNameInQuery(const std::string &query,
                                        const std::string &oldName,
                                        const std::string &newName,
                                        const std::string &oldAlias);
    void generateBatchCombinationsRecursive(
        const std::map<std::string, std::vector<std::string>> &tableGroups,
        std::map<std::string, std::string> currentCombination,
        std::map<std::string, std::vector<std::string>>::const_iterator currentGroup,
        std::vector<std::map<std::string, std::string>> &result);

    std::vector<std::map<std::string, std::string>> generateBatchCombinations(
        const std::map<std::string, std::vector<std::string>> &tableGroups);
    std::shared_ptr<Table> processBatchedQuery(const hsql::SelectStatement *stmt, const std::string &query);
    unionV copyUnionValue(const unionV &value, ColumnType type);

private:
    std::shared_ptr<StorageManager> storage_;
    std::unique_ptr<PlanBuilder> plan_builder_;

    void validateSelectStatement(const hsql::SelectStatement *stmt);
};