#pragma once
#include <hsql/SQLParser.h>
#include <memory>
#include "DataHandling/Table.hpp"

class QueryExecutor;

class SubqueryHandler
{
public:
    SubqueryHandler(std::shared_ptr<QueryExecutor> executor);

    std::shared_ptr<Table> handle(const hsql::SelectStatement *subquery);
    std::shared_ptr<Table> handleExists(const hsql::SelectStatement *subquery);
    std::shared_ptr<Table> handleIn(const hsql::Expr *expr);

private:
    std::shared_ptr<QueryExecutor> executor;
};