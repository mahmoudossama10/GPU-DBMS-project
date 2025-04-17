#pragma once
#include <memory>
#include <hsql/SQLParser.h>
#include "../DataHandling/Table.hpp"
#include "../DataHandling/StorageManager.hpp"

class ExecutionPlan
{
public:
    virtual ~ExecutionPlan() = default;
    virtual std::shared_ptr<Table> execute() = 0;
};

class PlanBuilder
{
public:
    explicit PlanBuilder(std::shared_ptr<StorageManager> storage);
    std::unique_ptr<ExecutionPlan> build(const hsql::SelectStatement *stmt);

private:
    std::shared_ptr<StorageManager> storage_;

    std::unique_ptr<ExecutionPlan> buildScanPlan(const hsql::TableRef *table);
    std::unique_ptr<ExecutionPlan> buildFilterPlan(std::unique_ptr<ExecutionPlan> input,
                                                   const hsql::Expr *where);

    // Add to existing PlanBuilder class
    std::unique_ptr<ExecutionPlan> buildProjectPlan(
        std::unique_ptr<ExecutionPlan> input,
        const std::vector<hsql::Expr *> &select_list);
};