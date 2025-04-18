#include "../../include/QueryProcessing/PlanBuilder.hpp"
#include "../../include/Operations/Filter.hpp"
#include "../../include/Operations/Project.hpp"
#include "../../include/Operations/Aggregator.hpp"
#include "../../include/Operations/OrderBy.hpp"
#include "../../include/Operations/Join.hpp"
#include "Utilities/ErrorHandling.hpp"
#include <iostream>
namespace
{
    bool isSelectAll(const std::vector<hsql::Expr *> *selectList)
    {
        // First, handle the simple "SELECT *" case, but only look at non‐function items:
        {
            int nonFuncCount = 0;
            const hsql::Expr *onlyExpr = nullptr;
            for (auto *expr : *selectList)
            {
                if (expr->type == hsql::kExprFunctionRef)
                {
                    // ignore function calls entirely
                    continue;
                }
                nonFuncCount++;
                if (nonFuncCount == 1)
                {
                    onlyExpr = expr;
                }
            }
            if (nonFuncCount <= 1 || onlyExpr->type == hsql::kExprStar)
            {
                return true;
            }
        }

        // Next, handle "SELECT table.*" (or mixed lists) by skipping functions again:
        for (auto *expr : *selectList)
        {
            if (expr->type == hsql::kExprFunctionRef)
            {
                continue; // ignore functions
            }
            if (expr->type == hsql::kExprStar)
            {
                return true;
            }
        }

        return false;
    }
    bool hasAggregates(const std::vector<hsql::Expr *> &select_list)
    {
        for (auto *expr : select_list)
        {
            if (expr->type == hsql::kExprFunctionRef)
                return true;
        }
        return false;
    }
}

class TableScanPlan : public ExecutionPlan
{
public:
    TableScanPlan(std::shared_ptr<StorageManager> storage,
                  const std::string &table_name,
                  const std::string &alias = "")
        : storage_(storage), table_name_(table_name), alias_(alias) {}

    std::shared_ptr<Table> execute() override
    {
        return std::make_shared<Table>(storage_->getTable(table_name_));
    }

    const std::string &getAlias() const { return alias_; }

private:
    std::shared_ptr<StorageManager> storage_;
    std::string table_name_;
    std::string alias_;
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

    if (hasAggregates(*(stmt->selectList)))
    {
        plan = buildAggregatePlan(std::move(plan), *(stmt->selectList));
    }

    // Only add ProjectPlan if needed (not SELECT *)
    if (!isSelectAll(stmt->selectList))
    {
        std::cout << "Why tf are you even" << '\n';
        plan = buildProjectPlan(std::move(plan), *(stmt->selectList));
    }

    if (stmt->order && !stmt->order->empty())
    {
        plan = buildOrderByPlan(std::move(plan), *stmt->order);
    }

    return plan;
}

std::unique_ptr<ExecutionPlan> PlanBuilder::buildProjectPlan(
    std::unique_ptr<ExecutionPlan> input,
    const std::vector<hsql::Expr *> &select_list)
{
    return std::make_unique<ProjectPlan>(std::move(input), select_list);
}

std::unique_ptr<ExecutionPlan> PlanBuilder::buildAggregatePlan(
    std::unique_ptr<ExecutionPlan> input,
    const std::vector<hsql::Expr *> &select_list)
{
    return std::make_unique<AggregatorPlan>(std::move(input), select_list);
}

std::unique_ptr<ExecutionPlan> PlanBuilder::buildOrderByPlan(
    std::unique_ptr<ExecutionPlan> input,
    const std::vector<hsql::OrderDescription *> &order_exprs)
{
    return std::make_unique<OrderByPlan>(std::move(input), order_exprs);
}

std::unique_ptr<ExecutionPlan> PlanBuilder::buildScanPlan(const hsql::TableRef *table)
{
    switch (table->type)
    {
    case hsql::kTableName:
    {
        std::string alias = table->alias ? std::string(table->alias->name) : "";
        return std::make_unique<TableScanPlan>(storage_, table->name, alias);
    }

    case hsql::kTableCrossProduct:
    {
        if (!table->list || table->list->size() != 2)
        {
            throw SemanticError("Unsupported cross product specification");
        }

        auto left = buildScanPlan(table->list->at(0));
        auto right = buildScanPlan(table->list->at(1));

        std::string left_alias, right_alias;

        if (auto *left_scan = dynamic_cast<TableScanPlan *>(left.get()))
        {
            left_alias = left_scan->getAlias();
        }
        if (auto *right_scan = dynamic_cast<TableScanPlan *>(right.get()))
        {
            right_alias = right_scan->getAlias();
        }

        return std::make_unique<JoinPlan>(
            std::move(left),
            std::move(right),
            left_alias,
            right_alias);
    }

    default:
        throw SemanticError("Unsupported table reference type");
    }
}

std::unique_ptr<ExecutionPlan> PlanBuilder::buildFilterPlan(
    std::unique_ptr<ExecutionPlan> input,
    const hsql::Expr *where)
{
    return std::make_unique<FilterPlan>(std::move(input), where);
}