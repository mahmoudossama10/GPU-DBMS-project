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
#include <algorithm>

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

    bool selectListNeedsProjection(const std::vector<hsql::Expr *> &selectList)
    {
        // If * present, never project
        for (auto *expr : selectList)
            if (expr->type == hsql::kExprStar)
                return false;

        // If *table present, also never project
        for (auto *expr : selectList)
        {
            if (expr->type == hsql::kExprColumnRef && expr->name && std::strcmp(expr->name, "*") == 0)
                return false;
        }

        // If any non-aggregate expression, projection is required
        for (auto *expr : selectList)
            if (expr->type != hsql::kExprFunctionRef)
                return true;

        // All are aggregates: NO projection needed
        return false;
    }

    // Helper function to convert string to unionV based on column type
    unionV stringToUnionValue(const std::string &value, ColumnType type)
    {
        unionV result;

        switch (type)
        {
        case ColumnType::STRING:
            result.s = new std::string(value);
            break;

        case ColumnType::INTEGER:
            result.i = std::stoll(value);
            break;

        case ColumnType::DOUBLE:
            result.d = std::stod(value);
            break;

        case ColumnType::DATETIME:
        {
            // Parse datetime string and create a dateTime struct
            // This is a simplified implementation - you may need more sophisticated parsing
            result.t = new dateTime();
            // Assuming format: YYYY-MM-DD HH:MM:SS
            sscanf(value.c_str(), "%hu-%hu-%hu %hhu:%hhu:%hhu",
                   &result.t->year, &result.t->month, &result.t->day,
                   &result.t->hour, &result.t->minute, &result.t->second);
            break;
        }

        default:
            throw std::runtime_error("Unknown column type");
        }

        return result;
    }

    // Helper function to get string representation from unionV
    std::string unionValueToString(const unionV &value, ColumnType type)
    {
        switch (type)
        {
        case ColumnType::STRING:
            return *(value.s);

        case ColumnType::INTEGER:
            return std::to_string(value.i);

        case ColumnType::DOUBLE:
            return std::to_string(value.d);

        case ColumnType::DATETIME:
        {
            char buffer[64];
            snprintf(buffer, sizeof(buffer), "%04hu-%02hu-%02hu %02hhu:%02hhu:%02hhu",
                     value.t->year, value.t->month, value.t->day,
                     value.t->hour, value.t->minute, value.t->second);
            return std::string(buffer);
        }

        default:
            throw std::runtime_error("Unknown column type");
        }
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
        auto table = std::make_shared<Table>(storage_->getTable(table_name_));
        if (!alias_.empty())
        {
            table->setAlias(alias_);
        }
        return table;
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

void collectConditions(const hsql::Expr *expr, std::vector<const hsql::Expr *> &conditions)
{
    if (expr == nullptr)
        return;

    if (expr->opType == hsql::kOpAnd || expr->opType == hsql::kOpOr)
    {
        collectConditions(expr->expr, conditions);
        collectConditions(expr->expr2, conditions);
    }
    else
    {
        conditions.push_back(expr);
    }
}

void collectInvolvedTables(const hsql::Expr *expr, std::unordered_set<std::string> &tables)
{
    if (!expr)
        return;

    if (expr->type == hsql::kExprColumnRef && expr->table != nullptr)
    {
        tables.insert(expr->table);
    }

    collectInvolvedTables(expr->expr, tables);
    collectInvolvedTables(expr->expr2, tables);

    if (expr->exprList)
    {
        for (const auto *subExpr : *expr->exprList)
        {
            collectInvolvedTables(subExpr, tables);
        }
    }
}

std::shared_ptr<Table> GPUJoinPlan::execute()
{
    if (tables_.empty())
    {
        throw SemanticError("No tables to join in GPU plan");
    }

    // Multi-table join with batched processing
    return gpu_manager_->executeBatchedJoin(tables_, where_clause_);
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

// std::unique_ptr<ExecutionPlan> PlanBuilder::build(const hsql::SelectStatement *stmt, const std::string &query)
// {
//     if (!stmt)
//     {
//         throw SemanticError("Invalid SELECT statement");
//     }

//     // First, process subqueries in the WHERE clause, if any
//     setExecutionMode(ExecutionMode::GPU);

//     hsql::Expr *processed_where = nullptr;
//     if (stmt->whereClause && hasSubquery(stmt->whereClause))
//     {
//         processed_where = processWhereWithSubqueries(stmt->whereClause);
//     }
//     else
//     {
//         processed_where = const_cast<hsql::Expr *>(stmt->whereClause);
//     }

//     // Check for subqueries in the FROM clause
//     bool has_subquery_in_from = hasSubqueryInTableRef(stmt->fromTable);

//     setExecutionMode(ExecutionMode::GPU);

//     // If using GPU and there are no complex subqueries, use GPU path
//     if (execution_mode_ == ExecutionMode::GPU && !has_subquery_in_from)
//     {

//         std::unique_ptr<ExecutionPlan> plan;

//         if (stmt->fromTable->type == hsql::kTableName)
//         {
//             // Single table only — safe to scan directly
//             auto plan = buildScanPlan(stmt->fromTable);
//             if (processed_where)
//             {
//                 plan = buildFilterPlan(std::move(plan), processed_where);
//             }

//             // if (hasAggregates(*(stmt->selectList)))
//             // {
//             //     plan = buildAggregatePlan(std::move(plan), *(stmt->selectList));
//             // }

//             // Only create ProjectPlan if needed
//             if (!isSelectAll(stmt->selectList) && selectListNeedsProjection(*(stmt->selectList)))
//             {
//                 plan = buildProjectPlan(std::move(plan), *(stmt->selectList));
//             }

//             // if (stmt->order && !stmt->order->empty())
//             // {
//             //     plan = buildOrderByPlan(std::move(plan), *stmt->order);
//             // }

//             return plan;
//         }
//         else
//         {
//             auto plan = buildGPUScanPlan(stmt->fromTable, processed_where);

//             // if (hasAggregates(*(stmt->selectList)))
//             // {
//             //     plan = buildAggregatePlan(std::move(plan), *(stmt->selectList));
//             // }

//             // Only create ProjectPlan if needed
//             if (!isSelectAll(stmt->selectList) && selectListNeedsProjection(*(stmt->selectList)))
//             {
//                 plan = buildProjectPlan(std::move(plan), *(stmt->selectList));
//             }

//             // if (stmt->order && !stmt->order->empty())
//             // {
//             //     plan = buildOrderByPlan(std::move(plan), *stmt->order);
//             // }
//             return plan;
//         }
//     }
//     else
//     {
//         // CPU path
//         auto plan = buildScanPlan(stmt->fromTable);

//         if (processed_where)
//         {
//             plan = buildFilterPlan(std::move(plan), processed_where);
//         }

//         // if (hasAggregates(*(stmt->selectList)))
//         // {
//         //     plan = buildAggregatePlan(std::move(plan), *(stmt->selectList));
//         // }

//         if (!isSelectAll(stmt->selectList) && selectListNeedsProjection(*(stmt->selectList)))
//         {
//             plan = buildProjectPlan(std::move(plan), *(stmt->selectList));
//         }

//         // if (stmt->order && !stmt->order->empty())
//         // {
//         //     plan = buildOrderByPlan(std::move(plan), *stmt->order);
//         // }
//         return plan;
//     }
// }

std::unique_ptr<ExecutionPlan> PlanBuilder::build(const hsql::SelectStatement *stmt, const std::string &query)
{
    if (!stmt)
    {
        throw SemanticError("Invalid SELECT statement");
    }

    // Initialize DuckDB
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);

    // Get query plan from DuckDB
    con.Query("CREATE TABLE employees AS SELECT * FROM read_csv_auto('./data/input/employees_new.csv');");
    con.Query("CREATE TABLE departments AS SELECT * FROM read_csv_auto('./data/input/departments_new.csv');");
    con.Query("CREATE TABLE projects AS SELECT * FROM read_csv_auto('./data/input/projects_new.csv');");

    auto duckdb_plan = con.ExtractPlan(query);

    // Convert DuckDB plan to our execution plan tree
    return convertDuckDBPlanToExecutionPlan(std::move(duckdb_plan));
}

std::unique_ptr<ExecutionPlan> PlanBuilder::convertDuckDBPlanToExecutionPlan(
    std::unique_ptr<duckdb::LogicalOperator> duckdb_plan)
{
    std::cout << duckdb_plan->ToString() << std::endl;

    // Base case for recursion
    if (!duckdb_plan)
    {
        return nullptr;
    }

    // First process children (postorder traversal)
    std::vector<std::unique_ptr<ExecutionPlan>> children;
    for (auto &child : duckdb_plan->children)
    {
        children.push_back(convertDuckDBPlanToExecutionPlan(std::move(child)));
    }

    // Then create our node
    auto node_type = duckdb_plan->type;
    std::unique_ptr<ExecutionPlan> plan;

    switch (node_type)
    {
    case duckdb::LogicalOperatorType::LOGICAL_GET:
    {
        auto *tablelol = dynamic_cast<duckdb::LogicalGet *>(duckdb_plan.get());
        std::cout << tablelol->GetTable();
        // std::string table_name = tablelol->; // Actual table name
        // plan = std::make_unique<TableScanPlan>(storage_, scan->table->name, alias);
        break;
    }
    // case duckdb::PhysicalOperatorType::FILTER:
    // {
    //     auto filter = static_cast<duckdb::PhysicalFilter *>(duckdb_plan.get());
    //     // Need to convert DuckDB's filter expression to HSQL
    //     hsql::Expr *where = convertDuckDBExprToHSQL(filter->expressions[0].get());
    //     plan = std::make_unique<FilterPlan>(std::move(children[0]), where);
    //     break;
    // }
    // case duckdb::PhysicalOperatorType::PROJECTION:
    // {
    //     auto proj = static_cast<duckdb::PhysicalProjection *>(duckdb_plan.get());
    //     // Convert projection expressions
    //     std::vector<hsql::Expr *> select_list;
    //     for (auto &expr : proj->expressions)
    //     {
    //         select_list.push_back(convertDuckDBExprToHSQL(expr.get()));
    //     }
    //     plan = std::make_unique<ProjectPlan>(std::move(children[0]), select_list);
    //     break;
    // }
    // Add other operator types as needed
    default:
        throw SemanticError("Unsupported DuckDB operator type");
    }

    return plan;
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
    case hsql::kTableSelect:
    {
        // Handle subquery in table reference
        std::string alias = table->alias ? std::string(table->alias->name) : "subquery";
        result.emplace_back(alias, alias); // Use alias for both name and alias
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

        // Check if this is a subquery result
        // if (ref.first == ref.second && ref.first != "" && hasSubqueryInTableRef(table))
        // {
        //     // This is likely a subquery reference, process it
        //     auto subquery_table = processSubqueryInFrom(table);
        //     if (subquery_table)
        //     {
        //         tables.push_back(subquery_table);
        //         tables.back()->setAlias(alias);
        //         table_names.push_back(alias);
        //         continue;
        //     }
        // }

        // Regular table
        tables.push_back(std::make_shared<Table>(storage_->getTable(ref.first)));
        tables.back()->setAlias(alias);
        table_names.push_back(ref.first);
    }

    // Create GPU join plan that handles filter conditions

    return std::make_unique<GPUJoinPlan>(tables, table_names, where, gpu_manager_);
}

// std::shared_ptr<Table> PlanBuilder::processSubqueryInFrom(const hsql::TableRef *table)
// {
//     if (table->type == hsql::kTableSelect)
//     {
//         // Process subquery directly
//         auto subquery_plan = build(table->select);
//         return subquery_plan->execute();
//     }
//     else if (table->type == hsql::kTableCrossProduct && table->list)
//     {
//         // Check if any table in cross product is a subquery
//         for (auto *t : *table->list)
//         {
//             if (t->type == hsql::kTableSelect)
//             {
//                 // Process just this subquery
//                 auto subquery_plan = build(t->select);
//                 return subquery_plan->execute();
//             }
//         }
//     }
//     return nullptr;
// }

std::unique_ptr<ExecutionPlan> PlanBuilder::buildProjectPlan(
    std::unique_ptr<ExecutionPlan> input,
    const std::vector<hsql::Expr *> &select_list)
{
    // Process any subqueries in the SELECT list
    std::vector<hsql::Expr *> processed_select_list = select_list;
    // for (auto &expr : processed_select_list)
    // {
    //     if (expr->type == hsql::kExprSelect)
    //     {
    //         // Process subquery and replace with a constant expression
    //         auto subquery_result = processSubqueryExpression(expr);
    //         // Replace the original expr with processed one if needed
    //         if (subquery_result)
    //         {
    //             expr = subquery_result;
    //         }
    //     }
    // }

    return std::make_unique<ProjectPlan>(std::move(input), processed_select_list);
}

// std::unique_ptr<ExecutionPlan> PlanBuilder::buildAggregatePlan(
//     std::unique_ptr<ExecutionPlan> input,
//     const std::vector<hsql::Expr *> &select_list)
// {
//     return std::make_unique<AggregatorPlan>(std::move(input), select_list);
// }

// std::unique_ptr<ExecutionPlan> PlanBuilder::buildOrderByPlan(
//     std::unique_ptr<ExecutionPlan> input,
//     const std::vector<hsql::OrderDescription *> &order_exprs)
// {
//     return std::make_unique<OrderByPlan>(std::move(input), order_exprs);
// }

std::unique_ptr<ExecutionPlan> PlanBuilder::buildScanPlan(const hsql::TableRef *table)
{
    if (!table)
    {
        throw SemanticError("Table reference is null");
    }

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

        // // Process the subquery by recursively calling build
        // std::unique_ptr<ExecutionPlan> subquery_plan = build(table->select);

        // // Execute the subquery plan to get the resulting table
        // std::shared_ptr<Table> subquery_result = subquery_plan->execute();

        // // Apply alias if specified
        // std::string alias = table->alias ? std::string(table->alias->name) : "subquery";
        // subquery_result->setAlias(alias);

        // // Return a plan that will yield the subquery result
        // return std::make_unique<SubqueryPlan>(subquery_result);
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
    // WHERE clause should already be processed for subqueries by this point
    return std::make_unique<FilterPlan>(std::move(input), where);
}

// // Process WHERE clauses that contain subqueries
// hsql::Expr *PlanBuilder::processWhereWithSubqueries(const hsql::Expr *expr)
// {
//     if (!expr)
//         return nullptr;

//     // If this is a subquery expression, process it and return its result
//     if (expr->type == hsql::kExprSelect)
//     {
//         return processSubqueryExpression(expr);
//     }

//     // Create a new expression with the same type and operation
//     hsql::Expr *result = new hsql::Expr(expr->type);
//     result->opType = expr->opType;

//     // Copy name if present
//     if (expr->name)
//     {
//         result->name = strdup(expr->name);
//     }
//     else
//     {
//         result->name = nullptr;
//     }

//     // Process left and right children recursively
//     if (expr->expr)
//     {
//         result->expr = processWhereWithSubqueries(expr->expr);
//     }

//     if (expr->expr2)
//     {
//         result->expr2 = processWhereWithSubqueries(expr->expr2);
//     }

//     // Process expression list if present
//     if (expr->exprList)
//     {
//         result->exprList = new std::vector<hsql::Expr *>();
//         for (auto *child : *expr->exprList)
//         {
//             result->exprList->push_back(processWhereWithSubqueries(child));
//         }
//     }

//     // Copy other fields
//     result->ival = expr->ival;
//     result->fval = expr->fval;
//     result->table = expr->table ? strdup(expr->table) : nullptr;

//     return result;
// }

// // Process a subquery expression and return its result as a literal expression
// hsql::Expr *PlanBuilder::processSubqueryExpression(const hsql::Expr *expr)
// {
//     if (!expr || expr->type != hsql::kExprSelect)
//     {
//         throw SemanticError("Expected a subquery expression");
//     }

//     // Recursively build and execute the subquery plan
//     std::unique_ptr<ExecutionPlan> subquery_plan = build(expr->select);
//     std::shared_ptr<Table> subquery_result = subquery_plan->execute();

//     // Create a literal expression based on the subquery result
//     hsql::Expr *result = nullptr;

//     // For scalar subquery, get the single value from the result
//     if (subquery_result->getSize() == 1 && subquery_result->getHeaders().size() == 1)
//     {
//         // Get the column name (first and only header)
//         const std::string &columnName = subquery_result->getHeaders()[0];

//         // Get the column type
//         ColumnType type = subquery_result->getColumnType(columnName);

//         // Get the first value using the appropriate accessor based on type
//         switch (type)
//         {
//         case ColumnType::INTEGER:
//         {
//             int64_t value = subquery_result->getInteger(columnName, 0);
//             result = new hsql::Expr(hsql::kExprLiteralInt);
//             result->ival = value;
//             break;
//         }
//         case ColumnType::DOUBLE:
//         {
//             double value = subquery_result->getDouble(columnName, 0);
//             result = new hsql::Expr(hsql::kExprLiteralFloat);
//             result->fval = value;
//             break;
//         }
//         case ColumnType::STRING:
//         {
//             std::string value = subquery_result->getString(columnName, 0);
//             result = new hsql::Expr(hsql::kExprLiteralString);
//             result->name = strdup(value.c_str());
//             break;
//         }
//         case ColumnType::DATETIME:
//         {
//             // For datetime, convert to string representation
//             const dateTime &dt = subquery_result->getDateTime(columnName, 0);
//             char buffer[64];
//             snprintf(buffer, sizeof(buffer), "%04hu-%02hu-%02hu %02hhu:%02hhu:%02hhu",
//                      dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second);

//             result = new hsql::Expr(hsql::kExprLiteralString);
//             result->name = strdup(buffer);
//             break;
//         }
//         default:
//             throw SemanticError("Unsupported column type in subquery result");
//         }
//     }
//     else
//     {
//         throw SemanticError("Unsupported subquery result format: must return a single column");
//     }

//     return result;
// }