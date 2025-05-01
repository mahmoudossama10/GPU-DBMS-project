#include "../../include/QueryProcessing/QueryExecutor.hpp"
#include "../../include/QueryProcessing/PlanBuilder.hpp"
#include "Utilities/ErrorHandling.hpp"

#include <hsql/util/sqlhelper.h>

QueryExecutor::QueryExecutor(std::shared_ptr<StorageManager> storage)
    : storage_(storage),
      plan_builder_(std::make_unique<PlanBuilder>(storage)) {}

std::shared_ptr<Table> QueryExecutor::execute(const std::string &query)
{
    hsql::SQLParserResult result;
    hsql::SQLParser::parse(query, &result);
    // hsql::printStatementInfo(result.getStatement(0));
    if (!result.isValid())
    {
        throw SyntaxError("Parse error: " + std::string(result.errorMsg()));
    }

    if (result.size() != 1)
    {
        throw SyntaxError("Only single-statement queries supported");
    }

    const auto *stmt = result.getStatement(0);
    if (stmt->type() != hsql::kStmtSelect)
    {
        throw SyntaxError("Only SELECT statements supported");
    }

    return execute(static_cast<const hsql::SelectStatement *>(stmt), query);
}

std::shared_ptr<Table> QueryExecutor::execute(const hsql::SelectStatement *stmt, const std::string &query)
{
    validateSelectStatement(stmt);
    return plan_builder_->build(stmt, query)->execute();
}

// Update validateSelectStatement
void QueryExecutor::validateSelectStatement(const hsql::SelectStatement *stmt)
{
    if (!stmt->fromTable)
    {
        throw SemanticError("Missing FROM clause");
    }

    // Validate projection list
    // for (const auto *expr : *(stmt->selectList))
    // {
    //     if (expr->type == hsql::kExprFunctionRef)
    //     {
    //         throw SemanticError("Aggregate functions not yet supported");
    //     }
    // }
}