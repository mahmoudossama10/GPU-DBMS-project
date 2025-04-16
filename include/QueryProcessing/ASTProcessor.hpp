#pragma once
#include <memory>
#include <hsql/SQLParser.h>
#include <hsql/sql/SelectStatement.h>

class ASTProcessor
{
public:
    struct ParsedQuery
    {
        std::vector<std::string> select_columns;
        std::vector<std::string> from_tables;
        const hsql::Expr *where_clause = nullptr;
        std::vector<hsql::OrderDescription *> order_by;
        hsql::GroupByDescription *group_by = nullptr;
        const hsql::SelectStatement *subquery = nullptr;
    };

    static ParsedQuery parse(const std::string &query);

private:
    static void processSelectColumns(const hsql::SelectStatement *stmt, ParsedQuery &result);
    static void processTableRef(const hsql::TableRef *table, ParsedQuery &result);
    static void processExpression(const hsql::Expr *expr);
};