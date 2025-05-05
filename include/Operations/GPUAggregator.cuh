#pragma once

#include "../../include/DataHandling/Table.hpp"
#include "../../include/QueryProcessing/PlanBuilder.hpp"
#include <hsql/util/sqlhelper.h>
#include <memory>
#include <vector>
#include <string>

class GPUAggregatorPlan : public ExecutionPlan {
public:
    GPUAggregatorPlan(std::unique_ptr<ExecutionPlan> input, const std::vector<hsql::Expr*>& select_list);
    std::shared_ptr<Table> execute() override;

private:
    struct AggregateOp {
        std::string function_name;
        std::string column_name;
        std::string alias;
        bool is_distinct;
        size_t column_index;
    };

    std::vector<AggregateOp> parseAggregates(const std::vector<hsql::Expr*>& select_list, const Table& table);
    std::shared_ptr<Table> aggregateTableGPU(const Table& table, const std::vector<AggregateOp>& aggregates);

    std::unique_ptr<ExecutionPlan> input_;
    std::vector<hsql::Expr*> select_list_;
};