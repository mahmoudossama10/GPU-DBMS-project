#include "../../include/QueryProcessing/PlanBuilder.hpp"
#include "../../include/Operations/Filter.hpp"
#include "../../include/Operations/Project.hpp"

#include "Utilities/ErrorHandling.hpp"
#include <iostream>
namespace
{
    bool isSelectAll(const std::vector<hsql::Expr *> *selectList)
    {
        // SELECT * case
        if (selectList->size() == 1 && (*selectList)[0]->type == hsql::kExprStar)
        {
            return true;
        }

        // SELECT table.* case
        for (const auto *expr : *selectList)
        {
            if (expr->type == hsql::kExprStar)
            {
                return true;
            }
        }

        return false;
    }
}

class TableScanPlan : public ExecutionPlan
{
public:
    TableScanPlan(std::shared_ptr<StorageManager> storage,
                  const std::string &table_name)
        : storage_(storage), table_name_(table_name) {}

    std::shared_ptr<Table> execute() override
    {
        return std::make_shared<Table>(storage_->getTable(table_name_));
    }

private:
    std::shared_ptr<StorageManager> storage_;
    std::string table_name_;
};

PlanBuilder::PlanBuilder(std::shared_ptr<StorageManager> storage)
    : storage_(storage) {}

std::unique_ptr<ExecutionPlan> PlanBuilder::build(const hsql::SelectStatement *stmt)
{
    auto plan = buildScanPlan(stmt->fromTable);

    // Apply WHERE clause if present
    if (stmt->whereClause)
    {

        plan = buildFilterPlan(std::move(plan), stmt->whereClause);
    }

    // Only add ProjectPlan if needed (not SELECT *)
    if (!isSelectAll(stmt->selectList))
    {
        plan = buildProjectPlan(std::move(plan), *(stmt->selectList));
    }

    return plan;
}

std::unique_ptr<ExecutionPlan> PlanBuilder::buildProjectPlan(
    std::unique_ptr<ExecutionPlan> input,
    const std::vector<hsql::Expr *> &select_list)
{
    return std::make_unique<ProjectPlan>(std::move(input), select_list);
}

std::unique_ptr<ExecutionPlan> PlanBuilder::buildScanPlan(const hsql::TableRef *table)
{
    if (table->type != hsql::kTableName)
    {
        throw SemanticError("Only direct table scans supported in minimal implementation");
    }
    return std::make_unique<TableScanPlan>(storage_, table->name);
}

std::unique_ptr<ExecutionPlan> PlanBuilder::buildFilterPlan(
    std::unique_ptr<ExecutionPlan> input,
    const hsql::Expr *where)
{
    return std::make_unique<FilterPlan>(std::move(input), where);
}