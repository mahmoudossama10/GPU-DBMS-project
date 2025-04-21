#include "../../include/QueryProcessing/PlanBuilder.hpp"
#include "../../include/Operations/Filter.hpp"
#include "../../include/Operations/Project.hpp"
#include "../../include/Operations/Aggregator.hpp"
#include "../../include/Operations/OrderBy.hpp"
#include "../../include/Operations/Join.hpp"
#include "Utilities/ErrorHandling.hpp"
#include <iostream>
#include <cstring>
#include <hsql/util/sqlhelper.h>

namespace
{
    bool isInteger(const std::string &s)
    {
        char *end;
        std::strtol(s.c_str(), &end, 10);
        return *end == '\0';
    }

    bool isFloat(const std::string &s)
    {
        char *end;
        std::strtod(s.c_str(), &end);
        return *end == '\0';
    }

    bool isSelectAll(const std::vector<hsql::Expr *> *selectList)
    {
        if (!selectList || selectList->empty())
        {
            return false;
        }

        // Check for "SELECT *" case
        for (auto *expr : *selectList)
        {
            if (expr->type == hsql::kExprStar)
            {
                return true;
            }
        }

        // Check for "SELECT table.*" case
        for (auto *expr : *selectList)
        {
            if (expr->type == hsql::kExprColumnRef &&
                expr->table != nullptr &&
                expr->name != nullptr &&
                strcmp(expr->name, "*") == 0)
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

    // Helper function to check if an expression contains a subquery
    bool hasSubquery(const hsql::Expr *expr)
    {
        if (!expr)
            return false;

        if (expr->type == hsql::kExprSelect)
        {
            return true;
        }

        // Check left child
        if (expr->expr && hasSubquery(expr->expr))
        {
            return true;
        }

        // Check right child
        if (expr->expr2 && hasSubquery(expr->expr2))
        {
            return true;
        }

        // Check for subqueries in operand list
        if (expr->exprList)
        {
            for (const hsql::Expr *operand : *expr->exprList)
            {
                if (hasSubquery(operand))
                {
                    return true;
                }
            }
        }

        return false;
    }
}

// Subquery plan that stores the result of a subquery execution
class SubqueryPlan : public ExecutionPlan
{
public:
    SubqueryPlan(std::shared_ptr<Table> result) : result_(result) {}

    std::shared_ptr<Table> execute() override
    {
        return result_;
    }

private:
    std::shared_ptr<Table> result_;
};

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

// Implementation of GPUJoinPlan
GPUJoinPlan::GPUJoinPlan(std::vector<std::shared_ptr<Table>> tables,
                         std::vector<std::string> table_names,
                         const hsql::Expr *where_clause,
                         std::shared_ptr<GPUManager> gpu_manager)
    : tables_(std::move(tables)),
      table_names_(std::move(table_names)),
      where_clause_(where_clause),
      gpu_manager_(gpu_manager) {}

std::shared_ptr<Table> GPUJoinPlan::execute()
{
    if (tables_.empty())
    {
        throw SemanticError("No tables to join in GPU plan");
    }

    // Handle single table case with potential filtering
    if (tables_.size() == 1)
    {
        auto &table = tables_[0];

        if (where_clause_)
        {
            // Apply GPU filter to single table
            auto mask = gpu_manager_->gpuFilterTable(*table, where_clause_);
            return gpu_manager_->applyFilter(*table, mask);
        }
        return table; // Return original table if no filter
    }

    // Multi-table join processing
    std::shared_ptr<Table> result = tables_[0];

    for (size_t i = 1; i < tables_.size(); i++)
    {
        result = gpu_manager_->executeJoin(result, tables_[i], where_clause_);
    }

    return result;
}

PlanBuilder::PlanBuilder(std::shared_ptr<StorageManager> storage, ExecutionMode mode)
    : storage_(storage), execution_mode_(mode)
{
    if (mode == ExecutionMode::GPU)
    {
        gpu_manager_ = std::make_shared<GPUManager>();
    }
}

void PlanBuilder::setExecutionMode(ExecutionMode mode)
{
    execution_mode_ = mode;
    if (mode == ExecutionMode::GPU && !gpu_manager_)
    {
        gpu_manager_ = std::make_shared<GPUManager>();
    }
}

std::unique_ptr<ExecutionPlan> PlanBuilder::build(const hsql::SelectStatement *stmt)
{

    // Check for subqueries in the FROM clause
    setExecutionMode(ExecutionMode::GPU);
    bool has_subquery = hasSubqueryInTableRef(stmt->fromTable);

    // If using GPU and there's no subquery in FROM, use GPU path
    if (execution_mode_ == ExecutionMode::GPU && !has_subquery)
    {
        // Use GPU for scan and filter in one operation
        if (stmt->whereClause)
        {
            auto plan = buildGPUScanPlan(stmt->fromTable, stmt->whereClause);

            // Continue with CPU operations for the rest of the pipeline
            if (hasAggregates(*(stmt->selectList)))
            {
                plan = buildAggregatePlan(std::move(plan), *(stmt->selectList));
            }

            if (!isSelectAll(stmt->selectList))
            {
                plan = buildProjectPlan(std::move(plan), *(stmt->selectList));
            }

            if (stmt->order && !stmt->order->empty())
            {
                plan = buildOrderByPlan(std::move(plan), *stmt->order);
            }

            return plan;
        }
        else
        {
            // Fall back to CPU path
            auto plan = buildScanPlan(stmt->fromTable);

            // Apply WHERE clause if present
            if (stmt->whereClause)
            {
                plan = buildFilterPlan(std::move(plan), stmt->whereClause);
            }

            // Continue with CPU operations for the rest of the pipeline
            if (hasAggregates(*(stmt->selectList)))
            {
                plan = buildAggregatePlan(std::move(plan), *(stmt->selectList));
            }

            if (!isSelectAll(stmt->selectList))
            {
                plan = buildProjectPlan(std::move(plan), *(stmt->selectList));
            }

            if (stmt->order && !stmt->order->empty())
            {
                plan = buildOrderByPlan(std::move(plan), *stmt->order);
            }

            return plan;
        }
    }
    else
    {

        // Fall back to CPU path
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
            plan = buildProjectPlan(std::move(plan), *(stmt->selectList));
        }

        if (stmt->order && !stmt->order->empty())
        {
            plan = buildOrderByPlan(std::move(plan), *stmt->order);
        }

        return plan;
    }
}

bool PlanBuilder::hasSubqueryInTableRef(const hsql::TableRef *table)
{
    if (!table)
        return false;

    switch (table->type)
    {
    case hsql::kTableSelect:
        return true;
    case hsql::kTableCrossProduct:
        if (table->list)
        {
            for (auto *t : *table->list)
            {
                if (hasSubqueryInTableRef(t))
                {
                    return true;
                }
            }
        }
        return false;
    default:
        return false;
    }
}

std::vector<std::pair<std::string, std::string>> PlanBuilder::extractTableReferences(const hsql::TableRef *table)
{
    std::vector<std::pair<std::string, std::string>> result;

    if (!table)
        return result;

    switch (table->type)
    {
    case hsql::kTableName:
    {
        std::string alias = table->alias ? std::string(table->alias->name) : "";
        result.emplace_back(table->name, alias);
        break;
    }
    case hsql::kTableCrossProduct:
        if (table->list)
        {
            for (auto *t : *table->list)
            {
                auto refs = extractTableReferences(t);
                result.insert(result.end(), refs.begin(), refs.end());
            }
        }
        break;
    default:
        break;
    }

    return result;
}

std::unique_ptr<ExecutionPlan> PlanBuilder::buildGPUScanPlan(const hsql::TableRef *table, const hsql::Expr *where)
{
    // Extract all table references
    auto table_refs = extractTableReferences(table);

    // Load all tables
    std::vector<std::shared_ptr<Table>> tables;
    std::vector<std::string> table_names;

    for (const auto &ref : table_refs)
    {
        std::string alias = ref.second.empty() ? ref.first : ref.second;

        tables.push_back(std::make_shared<Table>(storage_->getTable(ref.first)));
        tables.back()->setAlias(ref.second);
        table_names.push_back(ref.first);

        // Set alias if provided, otherwise use table name as alias
    }

    // Create GPU join plan that handles filter conditions
    return std::make_unique<GPUJoinPlan>(tables, table_names, where, gpu_manager_);
}

std::unique_ptr<ExecutionPlan> PlanBuilder::buildProjectPlan(
    std::unique_ptr<ExecutionPlan> input,
    const std::vector<hsql::Expr *> &select_list)
{
    // Process any subqueries in the SELECT list
    std::vector<hsql::Expr *> processed_select_list = select_list;
    for (auto &expr : processed_select_list)
    {
        if (expr->type == hsql::kExprSelect)
        {
            // Process subquery and replace with a constant expression
            auto subquery_result = processSubqueryExpression(expr);
            // TODO: Convert result to constant expression
        }
    }

    return std::make_unique<ProjectPlan>(std::move(input), processed_select_list);
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

    case hsql::kTableSelect:
    {
        // Handle subquery in FROM clause
        if (!table->select)
        {
            throw SemanticError("Subquery in FROM clause is null");
        }

        // Process the subquery first
        std::unique_ptr<ExecutionPlan> subquery_plan = build(table->select);

        // Execute the subquery plan to get the resulting table
        std::shared_ptr<Table> subquery_result = subquery_plan->execute();

        // Create a SubqueryPlan that returns this result
        std::string alias = table->alias ? std::string(table->alias->name) : "";
        if (alias.empty())
        {
            alias = "subquery"; // Default alias if none specified
        }

        // Return the SubqueryPlan that will yield the subquery result
        return std::make_unique<SubqueryPlan>(subquery_result);
    }

    case hsql::kTableCrossProduct:
    {
        if (!table->list || table->list->size() < 2)
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

        // Start with joining the first two tables
        auto join_plan = std::make_unique<JoinPlan>(
            std::move(left),
            std::move(right),
            left_alias,
            right_alias);

        // If there are more than 2 tables, join them one by one
        for (size_t i = 2; i < table->list->size(); i++)
        {
            auto next_table = buildScanPlan(table->list->at(i));
            std::string next_alias;

            if (auto *next_scan = dynamic_cast<TableScanPlan *>(next_table.get()))
            {
                next_alias = next_scan->getAlias();
            }

            join_plan = std::make_unique<JoinPlan>(
                std::move(join_plan),
                std::move(next_table),
                "", // No alias for intermediate result
                next_alias);
        }

        return join_plan;
    }

    default:
        throw SemanticError("Unsupported table reference type");
    }
}

std::unique_ptr<ExecutionPlan> PlanBuilder::buildFilterPlan(
    std::unique_ptr<ExecutionPlan> input,
    const hsql::Expr *where)
{
    // Process the WHERE clause to handle any subqueries
    hsql::Expr *processed_where = const_cast<hsql::Expr *>(where);

    // Check if the WHERE clause contains a subquery
    if (hasSubquery(where))
    {
        // Deep copy the WHERE clause to avoid modifying the original AST
        processed_where = processWhereWithSubqueries(where);
    }

    return std::make_unique<FilterPlan>(std::move(input), processed_where);
}

// New method to process subqueries in WHERE clause
hsql::Expr *PlanBuilder::processWhereWithSubqueries(const hsql::Expr *expr)
{
    if (!expr)
        return nullptr;

    // If this is a subquery expression, process it
    if (expr->type == hsql::kExprSelect)
    {
        return processSubqueryExpression(expr);
    }

    // Create a new expression with the same type and operation
    hsql::Expr *result = new hsql::Expr(expr->type);
    result->opType = expr->opType;
    result->name = expr->name;

    // Process left and right children recursively
    if (expr->expr)
    {
        result->expr = processWhereWithSubqueries(expr->expr);
    }

    if (expr->expr2)
    {
        result->expr2 = processWhereWithSubqueries(expr->expr2);
    }

    // Process expression list if present
    if (expr->exprList)
    {
        result->exprList = new std::vector<hsql::Expr *>();
        for (auto *child : *expr->exprList)
        {
            result->exprList->push_back(processWhereWithSubqueries(child));
        }
    }

    // Copy other fields as needed
    result->ival = expr->ival;
    result->fval = expr->fval;

    return result;
}

// New method to process a subquery expression
hsql::Expr *PlanBuilder::processSubqueryExpression(const hsql::Expr *expr)
{
    if (!expr || expr->type != hsql::kExprSelect)
    {
        throw SemanticError("Expected a subquery expression");
    }

    // Build and execute the subquery plan
    std::unique_ptr<ExecutionPlan> subquery_plan = build(expr->select);
    std::shared_ptr<Table> subquery_result = subquery_plan->execute();

    // Create a literal expression based on the subquery result
    hsql::Expr *result = nullptr;

    // For scalar subquery, get the single value from the result
    if (subquery_result->getSize() == 1 && subquery_result->getHeaders().size() == 1)
    {
        // Extract the single value from the result
        const std::string value = subquery_result->getData()[0][0];
        if (isInteger(value))
        {
            result = new hsql::Expr(hsql::kExprLiteralInt);
            result->ival = std::stoi(value);
        }
        else if (isFloat(value))
        {
            result = new hsql::Expr(hsql::kExprLiteralFloat);
            result->fval = std::stof(value);
        }
        else
        {
            result = new hsql::Expr(hsql::kExprLiteralString);
            result->name = new char[value.length() + 1];
            std::strcpy(result->name, value.c_str());
        }
    }
    else
    {
        throw SemanticError("Unsupported subquery type or result format");
    }

    return result;
}