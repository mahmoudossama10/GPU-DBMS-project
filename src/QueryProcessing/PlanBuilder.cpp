#include "../../include/QueryProcessing/PlanBuilder.hpp"
#include "Operations/Filter.hpp"
#include "Utilities/ErrorHandling.hpp"
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

class FilterPlan : public ExecutionPlan
{
public:
    FilterPlan(std::unique_ptr<ExecutionPlan> input,
               const hsql::Expr *condition)
        : input_(std::move(input)), condition_(condition) {}

    std::shared_ptr<Table> execute() override
    {
        auto table = input_->execute();
        return Filter::apply(table, condition_);
    }

private:
    std::unique_ptr<ExecutionPlan> input_;
    const hsql::Expr *condition_;
};

PlanBuilder::PlanBuilder(std::shared_ptr<StorageManager> storage)
    : storage_(storage) {}

std::unique_ptr<ExecutionPlan> PlanBuilder::build(const hsql::SelectStatement *stmt)
{
    auto plan = buildScanPlan(stmt->fromTable);

    if (stmt->whereClause)
    {
        plan = buildFilterPlan(std::move(plan), stmt->whereClause);
    }

    return plan;
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