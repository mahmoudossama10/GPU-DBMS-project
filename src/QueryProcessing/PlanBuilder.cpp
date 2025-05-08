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
#include <regex>

std::shared_ptr<Table> PlanBuilder::output_join_table = nullptr;
int PlanBuilder::joinPlansCount = 0;
ExecutionMode PlanBuilder::execution_mode_ = ExecutionMode::GPU;
namespace
{

    std::string simplifyCastExpressions(const std::string &input)
    {
        std::string result = input;

        // Regular expression to match CAST patterns
        // This pattern captures:
        // - The column name inside the CAST
        // - The comparison operator
        // - The value being compared against
        std::regex castPattern(R"(CAST\s*\(\s*([a-zA-Z0-9_]+)\s+AS\s+[a-zA-Z0-9_(),.\s]+\)\s*([<>=!]+)\s*([0-9.]+))");

        // Simplify by replacing with "column operator value"
        result = std::regex_replace(result, castPattern, "$1$2$3");

        return result;
    }

    std::string insertAndBetweenConditions(const std::string &input)
    {
        // First, remove "::TIMESTAMP" if found
        std::string processed_input = input;
        size_t timestamp_pos;
        while ((timestamp_pos = processed_input.find("::TIMESTAMP")) != std::string::npos)
        {
            processed_input.erase(timestamp_pos, 11); // "::TIMESTAMP" is 11 characters
        }

        // Regex to match expressions like: var = value, var >= value, etc.
        // But avoid matching date-time formats: yyyy-mm-dd hh:mm:ss
        std::regex cond_regex(R"((\s*[\w\.]+\s*(=|<|>|<=|>=|!=)\s*(?:(?!\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})[\w\.']+|\'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\')\s*))");

        std::sregex_iterator begin(processed_input.begin(), processed_input.end(), cond_regex);
        std::sregex_iterator end;

        std::ostringstream output;
        size_t last_pos = 0;

        for (auto it = begin; it != end; ++it)
        {
            auto match = *it;
            size_t match_start = match.position();
            size_t match_end = match_start + match.length();

            // Append anything before the match
            output << processed_input.substr(last_pos, match_start - last_pos);

            // Append the condition
            output << match.str();

            last_pos = match_end;

            // Peek at next match to see if something is between
            auto next_it = it;
            ++next_it;
            if (next_it != end)
            {
                std::string in_between = processed_input.substr(last_pos, next_it->position() - last_pos);
                if (in_between.find("AND") == std::string::npos && in_between.find("OR") == std::string::npos)
                {
                    output << " AND ";
                }
                else
                {
                    output << in_between;
                }
                last_pos = next_it->position();
            }
        }

        // Append remaining part of string
        output << processed_input.substr(last_pos);
        return output.str();
    }

    std::string insertAndBetweenComparisons(const std::string &input)
    {
        std::regex comparison(R"((\w+\s*(=|!=|<|>|<=|>=)\s*('[^']*'|[0-9]+)))");
        std::sregex_iterator iter(input.begin(), input.end(), comparison);
        std::sregex_iterator end;

        std::vector<std::string> comparisons;
        while (iter != end)
        {
            comparisons.push_back(iter->str());
            ++iter;
        }

        // Join with " AND "
        std::string result;
        for (size_t i = 0; i < comparisons.size(); i++)
        {
            result += comparisons[i];
            if (i < comparisons.size() - 1)
                result += " AND ";
        }

        return result;
    }

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
            result.i->value = std::stoll(value);
            break;

        case ColumnType::DOUBLE:
            result.d->value = std::stod(value);
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
            return std::to_string(value.i->value);

        case ColumnType::DOUBLE:
            return std::to_string(value.d->value);

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

class EmptyPlan : public ExecutionPlan
{
public:
    EmptyPlan(const hsql::SelectStatement *stmt, std::shared_ptr<StorageManager> storage_manager_)
        : stmt_(stmt), storage_manager_(storage_manager_) {}

    std::shared_ptr<Table> execute() override
    {

        std::vector<std::string> headers;
        bool has_wildcard = false;

        // First pass: collect explicit columns and detect wildcards
        for (const hsql::Expr *expr : *stmt_->selectList)
        {
            if (expr->type == hsql::kExprColumnRef)
            {
                std::string col_name = expr->table ? std::string(expr->table) + "." + expr->name : expr->name;
                headers.push_back(col_name);
            }
            else if (expr->type == hsql::kExprStar)
            {
                has_wildcard = true;
                headers.push_back("*"); // Temporary placeholder
            }
            else if (expr->alias)
            {
                headers.push_back(expr->alias);
            }
            else
            {
                headers.push_back("expr");
            }
        }

        // Handle wildcard expansion if needed
        if (has_wildcard)
        {
            std::vector<std::string> expanded_headers;

            for (const auto &header : headers)
            {
                if (header == "*")
                {
                    // Wildcard found - expand with all table columns
                    if (stmt_->fromTable)
                    {
                        try
                        {
                            Table &table = storage_manager_->getTable(stmt_->fromTable->name);
                            for (const auto &col : table.getHeaders())
                            {
                                expanded_headers.push_back(stmt_->fromTable->name + '.' + col);
                            }
                        }
                        catch (...)
                        {
                            throw std::runtime_error("Table not found for wildcard expansion: " +
                                                     std::string(stmt_->fromTable->name));
                        }
                    }
                    else
                    {
                        throw std::runtime_error("No table specified for wildcard expansion");
                    }
                }
                else
                {
                    expanded_headers.push_back(header);
                }
            }
            headers = std::move(expanded_headers);
        }

        // Create empty table structure
        std::unordered_map<std::string, std::vector<unionV>> empty_data;
        std::unordered_map<std::string, ColumnType> empty_types;

        for (const auto &header : headers)
        {
            empty_data[header] = {};
        }

        for (const auto &header : headers)
        {
            empty_types[header] = ColumnType::STRING; // Default type
        }

        return std::make_shared<Table>(
            "empty_result",
            headers,
            empty_data,
            empty_types);
    }

private:
    const hsql::SelectStatement *stmt_;
    std::shared_ptr<StorageManager> storage_manager_;
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

class PassPlan : public ExecutionPlan
{
public:
    PassPlan(std::unique_ptr<ExecutionPlan> input)
        : input_(std::move(input)) {}

    std::shared_ptr<Table> execute() override
    {

        return input_->execute();
    }

private:
    std::unique_ptr<ExecutionPlan> input_;
};

// Implementation of GPUJoinPlan
GPUJoinPlan::GPUJoinPlan(std::unique_ptr<ExecutionPlan> leftTable,
                         std::unique_ptr<ExecutionPlan> rightTable,
                         std::string where,
                         std::shared_ptr<GPUManager> gpu_manager)
    : leftTable_(std::move(leftTable)),
      rightTable_(std::move(rightTable)),
      whereString(where),
      gpu_manager_(gpu_manager) {}

GPUOrderByPlan::GPUOrderByPlan(std::shared_ptr<Table> input,
                               const std::vector<hsql::OrderDescription *> &order_exprs,
                               std::shared_ptr<GPUManager> gpu_manager)
    : input_(input), order_exprs_(order_exprs), gpu_manager_(gpu_manager) {}

std::shared_ptr<Table> GPUOrderByPlan::execute()
{

    return gpu_manager_->executeOrderBy(input_, order_exprs_);
}

GPUFilterPlan::GPUFilterPlan(std::unique_ptr<ExecutionPlan> input,
                             std::string where,
                             std::shared_ptr<GPUManager> gpu_manager)
    : input_(std::move(input)), whereString(where), gpu_manager_(gpu_manager) {}

std::shared_ptr<Table> GPUFilterPlan::execute()
{
    auto table = input_->execute();

    std::string sqlStatement = "SELECT * FROM dummy WHERE " + whereString;

    // Parse the SQL statement
    hsql::SQLParserResult result;
    hsql::SQLParser::parse(sqlStatement, &result);

    const auto *stmt = result.getStatement(0);

    auto selectStmt = static_cast<const hsql::SelectStatement *>(stmt);
    // Need to convert DuckDB's filter expression to HSQL

    const hsql::Expr *processed_where = const_cast<hsql::Expr *>(selectStmt->whereClause);

    where_clause_ = processed_where;

    return gpu_manager_->executeFilter(table, where_clause_);
}

GPUAggregatorPlan::GPUAggregatorPlan(std::shared_ptr<Table> input, const std::vector<hsql::Expr *> &select_list, std::shared_ptr<GPUManager> gpu_manager)
    : input_(input), select_list_(select_list), gpu_manager_(gpu_manager) {}

std::shared_ptr<Table> GPUAggregatorPlan::execute()
{
    if (!input_ || input_->getData().empty())
    {
        return input_;
    }
    return gpu_manager_->executeAggregate(input_, select_list_);
}

GPUJoinPlanMultipleTable::GPUJoinPlanMultipleTable(std::vector<std::unique_ptr<ExecutionPlan>> tables,
                                                   std::string where,
                                                   std::shared_ptr<GPUManager> gpu_manager)
    : tablesExecutionPlan_(std::move(tables)),
      whereString(where),
      gpu_manager_(gpu_manager)
{
}

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
    std::string sqlStatement = "SELECT * FROM dummy WHERE " + whereString;

    // Parse the SQL statement
    hsql::SQLParserResult result;
    hsql::SQLParser::parse(sqlStatement, &result);

    const auto *stmt = result.getStatement(0);

    auto selectStmt = static_cast<const hsql::SelectStatement *>(stmt);
    // Need to convert DuckDB's filter expression to HSQL

    auto leftTableData = leftTable_->execute();
    auto rightTableData = rightTable_->execute();

    // Multi-table join with batched processing
    // return gpu_manager_->executeBatchedJoin(tables_, where_clause_);
    return gpu_manager_->executeTwoTableJoinWithBinarySearch(leftTableData, rightTableData, selectStmt->whereClause);
}

std::shared_ptr<Table> GPUJoinPlanMultipleTable::execute()
{
    std::string sqlStatement = "SELECT * FROM dummy WHERE " + whereString;

    // Parse the SQL statement
    hsql::SQLParserResult result;
    hsql::SQLParser::parse(sqlStatement, &result);

    const auto *stmt = result.getStatement(0);

    auto selectStmt = static_cast<const hsql::SelectStatement *>(stmt);
    // Need to convert DuckDB's filter expression to HSQL

    for (auto &tempPlan : tablesExecutionPlan_)
    {
        tablesData_.push_back(tempPlan->execute());
    }

    // Multi-table join with batched processing
    // return gpu_manager_->executeBatchedJoin(tables_, where_clause_);
    return gpu_manager_->executeMultipleTableJoin(tablesData_, selectStmt->whereClause);
}

PlanBuilder::PlanBuilder(std::shared_ptr<StorageManager> storage, ExecutionMode mode)
    : storage_(storage)
{

    gpu_manager_ = std::make_shared<GPUManager>();
}

bool PlanBuilder::isSelectAll(const std::vector<hsql::Expr *> *selectList)
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

bool PlanBuilder::hasAggregates(const std::vector<hsql::Expr *> &select_list)
{
    for (auto *expr : select_list)
    {
        if (expr->type == hsql::kExprFunctionRef)
            return true;
    }
    return false;
}

bool PlanBuilder::hasOtherSelectNotAggregates(const std::vector<hsql::Expr *> &select_list)
{
    for (auto *expr : select_list)
    {
        if (expr->type != hsql::kExprFunctionRef)
            return true;
    }
    return false;
}

bool PlanBuilder::selectListNeedsProjection(const std::vector<hsql::Expr *> &selectList)
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
    std::vector<std::string> loaded_tables;

    // Dynamically create tables based on the query
    if (stmt->fromTable)
    {
        if (stmt->fromTable->type == hsql::kTableCrossProduct)
        {
            for (hsql::TableRef *tbl : *stmt->fromTable->list)
            {
                if (tbl->type == hsql::kTableName)
                {
                    std::string table_name = tbl->name;

                    // if (tbl->alias != nullptr)
                    //     table_name = std::string(tbl->name) + "_" + tbl->alias->name;
                    // else
                    table_name = std::string(tbl->name);
                    std::string file_name = "";

                    size_t pos = table_name.find("_batch");
                    if (pos != std::string::npos)
                    {
                        file_name = table_name.substr(0, pos);
                    }
                    else
                    {
                        file_name = table_name; // fallback to full name if "batch" not found
                    }

                    std::string create_query = "CREATE TABLE " + table_name + " AS SELECT * FROM read_csv_auto('" + storage_->inputDirectory + file_name + ".csv');";
                    con.Query(create_query);
                    loaded_tables.push_back(table_name);
                }
            }
        }
        else if (stmt->fromTable->type == hsql::kTableName)
        {

            std::string table_name = "";

            // if (stmt->fromTable->alias != nullptr)
            //     table_name = std::string(stmt->fromTable->name) + "_" + stmt->fromTable->alias->name;
            // else
            table_name = std::string(stmt->fromTable->name);
            std::string file_name = "";
            size_t pos = table_name.find("_batch");
            if (pos != std::string::npos)
            {
                file_name = table_name.substr(0, pos);
            }
            else
            {
                file_name = table_name; // fallback to full name if "batch" not found
            }

            std::string create_query = "CREATE TABLE " + table_name + " AS SELECT * FROM read_csv_auto('" + storage_->inputDirectory + file_name + ".csv');";
            con.Query(create_query);
            loaded_tables.push_back(table_name);
        }
    }

    std::string sql_query = query;

    // Create vectors to store table names, column names, and full patterns
    std::vector<std::string> table_names;
    std::vector<std::string> column_names;
    std::vector<std::string> full_patterns;
    std::vector<std::string> replaced_patterns;

    // Map to store table aliases (alias -> actual table name)
    std::map<std::string, std::string> alias_map;

    // 1. First, find table names and their aliases in the FROM clause
    std::regex from_pattern("FROM\\s+([a-zA-Z_][a-zA-Z0-9_]*)\\s+(?:AS\\s+)?([a-zA-Z_][a-zA-Z0-9_]*)");
    std::smatch from_match;
    std::string::const_iterator from_search(sql_query.cbegin());

    while (std::regex_search(from_search, sql_query.cend(), from_match, from_pattern))
    {
        std::string table_name = from_match[1];
        std::string alias = from_match[2];

        // Store alias mapping
        alias_map[alias] = table_name;

        // Move to the next match
        from_search = from_match.suffix().first;
    }

    // 2. Find additional tables in comma-separated list (e.g., "table1, table2 t2, table3 t3")
    std::regex comma_pattern(R"((?:FROM|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)(?:\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*))?(?=\s*,|\s+WHERE|$))");
    std::smatch comma_match;
    std::string::const_iterator comma_search(sql_query.cbegin());

    while (std::regex_search(comma_search, sql_query.cend(), comma_match, comma_pattern))
    {
        std::string table_name = comma_match[1];
        std::string alias = comma_match[2];

        // Store alias mapping if not already stored
        if (alias_map.find(alias) == alias_map.end())
        {
            alias_map[alias] = table_name;
        }

        // Move to the next match
        comma_search = comma_match.suffix().first;
    }

    // 3. Find patterns like table.column and replace them with quoted versions
    std::regex pattern("([a-zA-Z_][a-zA-Z0-9_]*)\\.(([a-zA-Z_][a-zA-Z0-9_]*))");
    std::smatch match;
    std::string::const_iterator search_start(sql_query.cbegin());

    // Vector to keep track of replacement positions
    std::vector<std::pair<size_t, std::pair<std::string, std::string>>> replacements;

    // Find all matches and store them
    while (std::regex_search(search_start, sql_query.cend(), match, pattern))
    {
        std::string alias = match[1];
        std::string column_name = match[2];
        std::string full_pattern = match[0];

        // Find the actual table name for this alias
        std::string actual_table = alias_map.find(alias) != alias_map.end() ? alias_map[alias] : alias;

        if (std::find(full_patterns.begin(), full_patterns.end(), full_pattern) == full_patterns.end())
        {
            table_names.push_back(actual_table);
            column_names.push_back(column_name);
            full_patterns.push_back(full_pattern);
            replaced_patterns.push_back("\"" + full_pattern + "\"");
        }

        // Calculate position in original string
        size_t pos = match.position() + (search_start - sql_query.cbegin());
        replacements.push_back({pos, {full_pattern, alias + "\.\"" + full_pattern + "\""}});

        // Move to the next match
        search_start = match.suffix().first;
    }

    // Sort replacements in reverse order to avoid position shifts
    std::sort(replacements.begin(), replacements.end(),
              [](const auto &a, const auto &b)
              { return a.first > b.first; });

    // Apply replacements
    std::string modified_query = sql_query;
    for (const auto &rep : replacements)
    {
        modified_query.replace(rep.first, rep.second.first.length(), rep.second.second);
    }

    // Print the results
    // std::cout << "Original Query:\n"
    //           << sql_query << "\n\n";
    // std::cout << "Modified Query:\n"
    //           << modified_query << "\n\n";

    for (int i = 0; i < table_names.size(); i++)
    {
        std::string rename_query = "ALTER TABLE " + table_names[i] + " RENAME COLUMN \"" + column_names[i] + "\" TO " + replaced_patterns[i] + ";";
        // std::cout << rename_query << '\n';
        con.Query(rename_query);
    }
    // std::cout << "Table Name -> Alias Mappings:\n";
    // for (const auto &[alias, table] : alias_map)
    // {
    //     std::cout << table << " -> " << alias << "\n";
    // }

    // // std::cout << "\nExtracted Table Names (based on aliases in dot notation):\n";
    // for (const auto &name : table_names)
    // {
    //     std::cout << name << "\n";
    // }

    // // std::cout << "\nExtracted Column Names:\n";
    // for (const auto &name : column_names)
    // {
    //     std::cout << name << "\n";
    // }

    // // std::cout << "\nFull Patterns (to be quoted):\n";
    // for (const auto &pattern : full_patterns)
    // {
    //     std::cout << pattern << "\n";
    // }

    auto duckdb_plan = con.ExtractPlan(modified_query);

    // Convert DuckDB plan to our execution plan tree
    int cnt = 0;
    return convertDuckDBPlanToExecutionPlan(stmt, std::move(duckdb_plan), cnt);
}

std::unique_ptr<ExecutionPlan> PlanBuilder::convertDuckDBPlanToExecutionPlan(const hsql::SelectStatement *stmt,
                                                                             std::unique_ptr<duckdb::LogicalOperator> duckdb_plan, int cnt)
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
        cnt++;
        if (child->type == duckdb::LogicalOperatorType::LOGICAL_CROSS_PRODUCT)
        {
            if (duckdb_plan->type == duckdb::LogicalOperatorType::LOGICAL_ANY_JOIN || duckdb_plan->type == duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN)
            {
                children.push_back(convertDuckDBPlanToExecutionPlan(stmt, std::move(child->children[0]), cnt));
                children.push_back(convertDuckDBPlanToExecutionPlan(stmt, std::move(child->children[1]), cnt));
            }
            else
            {
                children.push_back(convertDuckDBPlanToExecutionPlan(stmt, std::move(child), cnt));
            }
        }
        else
        {
            children.push_back(convertDuckDBPlanToExecutionPlan(stmt, std::move(child), cnt));
        }
    }

    // Then create our node
    auto node_type = duckdb_plan->type;
    std::unique_ptr<ExecutionPlan> plan;

    switch (node_type)
    {
    case duckdb::LogicalOperatorType::LOGICAL_GET:
    {
        auto *table_scan = dynamic_cast<duckdb::LogicalGet *>(duckdb_plan.get());
        auto params = table_scan->ParamsToString();
        std::string table_name = "";
        std::string alias = "";
        std::string filters = "";

        for (const auto &entry : params)
        {
            if (entry.first == "Table")
            {
                table_name = entry.second;
            }
            if (entry.first == "Filters")
            {
                if (entry.second.find("optional:") == std::string::npos)
                {
                    filters = simplifyCastExpressions(entry.second);
                    filters = insertAndBetweenConditions(filters);
                }
            }
        }
        auto table = stmt->fromTable;
        if (!table)
        {
            throw SemanticError("Table reference is null");
        }

        switch (table->type)
        {
        case hsql::kTableName:
        {
            table_name = std::string(table->name);
            alias = table->alias ? std::string(table->alias->name) : "";

            break;
        }
        case hsql::kTableCrossProduct:
        {

            for (size_t i = 0; i < table->list->size(); i++)
            {
                std::string temp_table_name = std::string(table->list->at(i)->name);
                if (table_name == temp_table_name)
                {
                    alias = table->list->at(i)->alias ? std::string(table->list->at(i)->alias->name) : "";
                }
            }
            break;
        }

        default:
            break;
        }
        // std::cout << table_name << alias << '\n';
        plan = std::make_unique<TableScanPlan>(storage_, table_name, alias);

        if (filters != "")
        {

            if (execution_mode_ == ExecutionMode::GPU)
            {
                // plan = buildGPUFilterPlan(std::move(plan), filters);
                plan = buildFilterPlan(std::move(plan), filters);
            }
            else
            {
                plan = buildFilterPlan(std::move(plan), filters);
            }
        }
        break;
    }
    case duckdb::LogicalOperatorType::LOGICAL_FILTER:
    {
        auto *table_filter = dynamic_cast<duckdb::LogicalFilter *>(duckdb_plan.get());
        auto params = table_filter->ParamsToString();
        std::string filter_condition_string = "";
        for (const auto &entry : params)
        {
            if (entry.first == "Expressions")
            {
                filter_condition_string = entry.second;
                filter_condition_string = simplifyCastExpressions(filter_condition_string);
                filter_condition_string = insertAndBetweenConditions(filter_condition_string);
            }
        }

        if (execution_mode_ == ExecutionMode::GPU)
        {
            // plan = buildGPUFilterPlan(std::move(children[0]), filter_condition_string);
            plan = buildFilterPlan(std::move(children[0]), filter_condition_string);
        }
        else
        {
            plan = buildFilterPlan(std::move(children[0]), filter_condition_string);
        }
        break;
    }
    case duckdb::LogicalOperatorType::LOGICAL_PROJECTION:
    {
        // auto *table_projection = dynamic_cast<duckdb::LogicalProjection *>(duckdb_plan.get());
        // auto params = table_projection->ParamsToString();

        // for (const auto &entry : params)
        // {
        //     std::cout << entry.first << ": " << entry.second << std::endl;
        // }

        plan = buildPassPlan(std::move(children[0]));

        // // Convert projection expressions
        // std::vector<hsql::Expr *> select_list;
        // for (auto &expr : proj->expressions)
        // {
        //     select_list.push_back(convertDuckDBExprToHSQL(expr.get()));
        // }
        // plan = std::make_unique<ProjectPlan>(std::move(children[0]), select_list);
        break;
    }
        // case duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
        // {
        //     auto *table_join = dynamic_cast<duckdb::LogicalComparisonJoin *>(duckdb_plan.get());
        //     auto params = table_join->ParamsToString();
        //     const auto &conditions = table_join->conditions;
        //     std::string filter_condition_string = "";
        //     for (const auto &entry : params)
        //     {
        //         if (entry.first == "Conditions")
        //         {
        //             filter_condition_string = entry.second;
        //         }
        //     }

        //     auto where = insertAndBetweenConditions(filter_condition_string);

        //     plan = buildGPUJoinPlan(std::move(children[0]), std::move(children[1]), where);
        //     break;
        // }

    case duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
    {
        auto *table_join = dynamic_cast<duckdb::LogicalComparisonJoin *>(duckdb_plan.get());
        auto params = table_join->ParamsToString();
        const auto &conditions = table_join->conditions;
        std::string filter_condition_string = "";
        for (const auto &entry : params)
        {
            if (entry.first == "Conditions")
            {
                filter_condition_string = entry.second;
            }
        }

        auto where = insertAndBetweenConditions(filter_condition_string);

        if (execution_mode_ == ExecutionMode::GPU)
        {
            if (children.size() > 2)
            {
                plan = buildGPUJoinPlanMultipleTable(std::move(children), where);
            }
            else
            {
                plan = buildGPUJoinPlan(std::move(children[0]), std::move(children[1]), where);
            }
            gpu_manager_->joinPlansCount++;
        }
        else
        {
            PlanBuilder::joinPlansCount++;
            plan = buildCPUJoinPlan(std::move(children), where);
        }
        break;
    }
    case duckdb::LogicalOperatorType::LOGICAL_ANY_JOIN:
    {
        auto *table_join = dynamic_cast<duckdb::LogicalAnyJoin *>(duckdb_plan.get());
        auto params = table_join->ParamsToString();
        std::string filter_condition_string = "";
        for (const auto &entry : params)
        {
            if (entry.first == "Condition")
            {
                filter_condition_string = entry.second;
            }
        }

        auto where = insertAndBetweenConditions(filter_condition_string);
        if (execution_mode_ == ExecutionMode::GPU)
        {
            if (children.size() > 2)
            {
                plan = buildGPUJoinPlanMultipleTable(std::move(children), where);
            }
            else
            {
                plan = buildGPUJoinPlan(std::move(children[0]), std::move(children[1]), where);
            }
            gpu_manager_->joinPlansCount++;
        }
        else
        {
            PlanBuilder::joinPlansCount++;
            plan = buildCPUJoinPlan(std::move(children), where);
        }
        break;
    }
    case duckdb::LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
    {
        auto *table_join = dynamic_cast<duckdb::LogicalCrossProduct *>(duckdb_plan.get());

        plan = buildCPUJoinPlan(std::move(children), "");

        break;
    }

    case duckdb::LogicalOperatorType::LOGICAL_EMPTY_RESULT:
    {
        auto *table_join = dynamic_cast<duckdb::LogicalCrossProduct *>(duckdb_plan.get());

        plan = buildEmptyPlan(stmt, storage_);

        break;
    }
    // Add other operator types as needed
    default:
        plan = buildPassPlan(std::move(children[0]));

        // plan = buildProjectPlan(std::move(children[0]), *(stmt->selectList));

        break;
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

// std::unique_ptr<ExecutionPlan> PlanBuilder::buildGPUScanPlan(const hsql::TableRef *table, const hsql::Expr *where)
// {
//     // Extract all table references
//     auto table_refs = extractTableReferences(table);

//     // Load all tables
//     std::vector<std::shared_ptr<Table>> tables;
//     std::vector<std::string> table_names;

//     for (const auto &ref : table_refs)
//     {
//         std::string alias = ref.second.empty() ? ref.first : ref.second;

//         // Check if this is a subquery result
//         // if (ref.first == ref.second && ref.first != "" && hasSubqueryInTableRef(table))
//         // {
//         //     // This is likely a subquery reference, process it
//         //     auto subquery_table = processSubqueryInFrom(table);
//         //     if (subquery_table)
//         //     {
//         //         tables.push_back(subquery_table);
//         //         tables.back()->setAlias(alias);
//         //         table_names.push_back(alias);
//         //         continue;
//         //     }
//         // }

//         // Regular table
//         tables.push_back(std::make_shared<Table>(storage_->getTable(ref.first)));
//         tables.back()->setAlias(alias);
//         table_names.push_back(ref.first);
//     }

//     // Create GPU join plan that handles filter conditions

//     return std::make_unique<GPUJoinPlan>(tables, table_names, where, gpu_manager_);
// }

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

std::shared_ptr<Table> PlanBuilder::buildProjectPlan(
    std::shared_ptr<Table> input,
    const std::vector<hsql::Expr *> &select_list)
{
    std::vector<hsql::Expr *> processed_select_list = select_list;

    auto plan = std::make_unique<ProjectPlan>(input, processed_select_list);
    return plan->execute();
}

std::unique_ptr<ExecutionPlan> PlanBuilder::buildCPUJoinPlan(
    std::vector<std::unique_ptr<ExecutionPlan>> inputs,
    std::string where)
{

    return std::make_unique<JoinPlan>(std::move(inputs), where);
}
std::unique_ptr<ExecutionPlan> PlanBuilder::buildPassPlan(
    std::unique_ptr<ExecutionPlan> input)
{
    return std::make_unique<PassPlan>(std::move(input));
}

std::unique_ptr<ExecutionPlan> PlanBuilder::buildEmptyPlan(
    const hsql::SelectStatement *stmt, std::shared_ptr<StorageManager> storage_manager_)
{
    return std::make_unique<EmptyPlan>(stmt, storage_manager_);
}

// std::unique_ptr<ExecutionPlan> PlanBuilder::buildAggregatePlan(
//     std::unique_ptr<ExecutionPlan> input,
//     const std::vector<hsql::Expr *> &select_list)
// {
//     return std::make_unique<AggregatorPlan>(std::move(input), select_list);
// }

std::shared_ptr<Table> PlanBuilder::buildOrderByPlan(
    std::shared_ptr<Table> input,
    const std::vector<hsql::OrderDescription *> &order_exprs)
{
    auto plan = std::make_unique<OrderByPlan>(input, order_exprs);
    auto result = plan->execute();
    return result;
}

std::shared_ptr<Table> PlanBuilder::buildGPUOrderByPlan(
    std::shared_ptr<Table> input,
    const std::vector<hsql::OrderDescription *> &order_exprs)
{
    auto plan = std::make_unique<GPUOrderByPlan>(input, order_exprs, gpu_manager_);
    auto result = plan->execute();
    return result;
}

std::shared_ptr<Table> PlanBuilder::buildGPUAggregatePlan(
    std::shared_ptr<Table> input,
    const std::vector<hsql::Expr *> &select_list)
{
    auto plan = std::make_unique<GPUAggregatorPlan>(input, select_list, gpu_manager_);
    auto result = plan->execute();
    return result;
}

std::shared_ptr<Table> PlanBuilder::buildCPUAggregatePlan(
    std::shared_ptr<Table> input,
    const std::vector<hsql::Expr *> &select_list)
{
    auto plan = std::make_unique<AggregatorPlan>(input, select_list);
    auto result = plan->execute();
    return result;
}

// std::unique_ptr<ExecutionPlan> PlanBuilder::buildScanPlan(const hsql::TableRef *table)
// {
//     if (!table)
//     {
//         throw SemanticError("Table reference is null");
//     }

//     switch (table->type)
//     {
//     case hsql::kTableName:
//     {
//         std::string alias = table->alias ? std::string(table->alias->name) : "";
//         return std::make_unique<TableScanPlan>(storage_, table->name, alias);
//     }

//     case hsql::kTableSelect:
//     {
//         // Handle subquery in FROM clause
//         if (!table->select)
//         {
//             throw SemanticError("Subquery in FROM clause is null");
//         }

//         // // Process the subquery by recursively calling build
//         // std::unique_ptr<ExecutionPlan> subquery_plan = build(table->select);

//         // // Execute the subquery plan to get the resulting table
//         // std::shared_ptr<Table> subquery_result = subquery_plan->execute();

//         // // Apply alias if specified
//         // std::string alias = table->alias ? std::string(table->alias->name) : "subquery";
//         // subquery_result->setAlias(alias);

//         // // Return a plan that will yield the subquery result
//         // return std::make_unique<SubqueryPlan>(subquery_result);
//     }

//     case hsql::kTableCrossProduct:
//     {
//         if (!table->list || table->list->size() < 2)
//         {
//             throw SemanticError("Unsupported cross product specification");
//         }

//         auto left = buildScanPlan(table->list->at(0));
//         auto right = buildScanPlan(table->list->at(1));

//         std::string left_alias, right_alias;

//         if (auto *left_scan = dynamic_cast<TableScanPlan *>(left.get()))
//         {
//             left_alias = left_scan->getAlias();
//         }
//         if (auto *right_scan = dynamic_cast<TableScanPlan *>(right.get()))
//         {
//             right_alias = right_scan->getAlias();
//         }

//         // Start with joining the first two tables
//         auto join_plan = std::make_unique<JoinPlan>(
//             std::move(left),
//             std::move(right),
//             left_alias,
//             right_alias);

//         // If there are more than 2 tables, join them one by one
//         for (size_t i = 2; i < table->list->size(); i++)
//         {
//             auto next_table = buildScanPlan(table->list->at(i));
//             std::string next_alias;

//             if (auto *next_scan = dynamic_cast<TableScanPlan *>(next_table.get()))
//             {
//                 next_alias = next_scan->getAlias();
//             }

//             join_plan = std::make_unique<JoinPlan>(
//                 std::move(join_plan),
//                 std::move(next_table),
//                 "", // No alias for intermediate result
//                 next_alias);
//         }

//         return join_plan;
//     }

//     default:
//         throw SemanticError("Unsupported table reference type");
//     }
// }

std::unique_ptr<ExecutionPlan> PlanBuilder::buildFilterPlan(
    std::unique_ptr<ExecutionPlan> input,
    const std::string where)
{
    // WHERE clause should already be processed for subqueries by this point
    return std::make_unique<FilterPlan>(std::move(input), where);
}

std::unique_ptr<ExecutionPlan> PlanBuilder::buildGPUJoinPlan(
    std::unique_ptr<ExecutionPlan> leftTable,
    std::unique_ptr<ExecutionPlan> rightTable,
    std::string where)
{
    // WHERE clause should already be processed for subqueries by this point
    return std::make_unique<GPUJoinPlan>(std::move(leftTable), std::move(rightTable), where, gpu_manager_);
}

std::unique_ptr<ExecutionPlan> PlanBuilder::buildGPUFilterPlan(
    std::unique_ptr<ExecutionPlan> input,
    std::string where)
{
    // WHERE clause should already be processed for subqueries by this point
    return std::make_unique<GPUFilterPlan>(std::move(input), where, gpu_manager_);
}

// std::unique_ptr<ExecutionPlan> PlanBuilder::buildGPUFilterPlan(
//     std::unique_ptr<ExecutionPlan> input,
//     std::string where)
// {
//     // WHERE clause should already be processed for subqueries by this point
//     return std::make_unique<GPUJoinPlan>(std::move(leftTable), std::move(rightTable), where, gpu_manager_);
// }

std::unique_ptr<ExecutionPlan> PlanBuilder::buildGPUJoinPlanMultipleTable(
    std::vector<std::unique_ptr<ExecutionPlan>> tables,
    std::string where)
{
    // WHERE clause should already be processed for subqueries by this point
    return std::make_unique<GPUJoinPlanMultipleTable>(std::move(tables), where, gpu_manager_);
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