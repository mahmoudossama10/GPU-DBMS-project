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
    std::shared_ptr<Table> execute(const hsql::SelectStatement *stmt);

private:
    std::shared_ptr<StorageManager> storage_;
    std::unique_ptr<PlanBuilder> plan_builder_;

    void validateSelectStatement(const hsql::SelectStatement *stmt);
};