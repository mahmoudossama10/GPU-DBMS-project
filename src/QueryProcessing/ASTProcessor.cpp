#include "QueryProcessing/ASTProcessor.hpp"
#include "Utilities/ErrorHandling.hpp"

ASTProcessor::ParsedQuery ASTProcessor::parse(const std::string &query)
{
    hsql::SQLParserResult result;
    hsql::SQLParser::parse(query, &result);

    if (!result.isValid())
    {
        throw SyntaxError("SQL parse error: " + std::string(result.errorMsg()));
    }

    if (result.size() != 1)
    {
        throw SyntaxError("Only single-statement queries are supported");
    }

    const hsql::SQLStatement *stmt = result.getStatement(0);
    if (stmt->type() != hsql::kStmtSelect)
    {
        throw SyntaxError("Only SELECT statements are supported");
    }

    ParsedQuery parsed;
    const auto *select = static_cast<const hsql::SelectStatement *>(stmt);

    processSelectColumns(select, parsed);
    processTableRef(select->fromTable, parsed);

    parsed.where_clause = select->whereClause;
    parsed.order_by = *(select->order);
    parsed.group_by = select->groupBy;

    return parsed;
}

void ASTProcessor::processSelectColumns(const hsql::SelectStatement *stmt, ParsedQuery &result)
{
    for (const hsql::Expr *expr : *stmt->selectList)
    {
        if (expr->type == hsql::kExprColumnRef)
        {
            result.select_columns.push_back(expr->name);
        }
        // Handle other expression types (aliases, functions, etc.)
    }
}

void ASTProcessor::processTableRef(const hsql::TableRef *table, ParsedQuery &result)
{
    switch (table->type)
    {
    case hsql::kTableName:
        result.from_tables.push_back(table->name);
        break;
    case hsql::kTableJoin:
        processTableRef(table->join->left, result);
        processTableRef(table->join->right, result);
        break;
    case hsql::kTableCrossProduct:
        for (const hsql::TableRef *tbl : *table->list)
        {
            processTableRef(tbl, result);
        }
        break;
    default:
        throw SemanticError("Unsupported table reference type");
    }
}