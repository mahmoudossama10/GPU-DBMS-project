#pragma once

#include "../../include/DataHandling/Table.hpp"
#include "../../include/QueryProcessing/PlanBuilder.hpp"
#include <hsql/util/sqlhelper.h>
#include <memory>
#include <vector>
#include <string>

class AggregatorPlan : public ExecutionPlan
{
public:
    AggregatorPlan(std::shared_ptr<Table> input, const std::vector<hsql::Expr *> &select_list);
    std::shared_ptr<Table> execute() override;

private:
    struct AggregateOp
    {
        std::string function_name;
        std::string column_name;
        std::string alias;
        bool is_distinct;
    };

    std::vector<AggregateOp> parseAggregates(const std::vector<hsql::Expr *> &select_list, const Table &table);
    std::shared_ptr<Table> aggregateTable(const Table &table, const std::vector<AggregateOp> &aggregates);

    // Helper to convert unionV to string
    std::string unionValueToString(const unionV &value, ColumnType type);

    std::shared_ptr<Table> input_;
    std::vector<hsql::Expr *> select_list_;
};