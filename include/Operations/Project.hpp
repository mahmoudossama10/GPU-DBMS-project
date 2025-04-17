#pragma once
#include "../DataHandling/Table.hpp"
#include "../QueryProcessing/PlanBuilder.hpp"
#include <hsql/SQLParser.h>
#include <memory>
#include <vector>

class ProjectPlan : public ExecutionPlan
{
public:
    ProjectPlan(std::unique_ptr<ExecutionPlan> input,
                const std::vector<hsql::Expr *> &select_list);
    std::shared_ptr<Table> execute() override;

private:
    std::unique_ptr<ExecutionPlan> input_;
    std::vector<hsql::Expr *> select_list_;

    std::vector<std::string> getColumnNames() const;
    std::shared_ptr<Table> processProjection(std::shared_ptr<Table> input) const;
    std::string evaluateExpression(const std::vector<std::string> &row,
                                   const hsql::Expr *expr,
                                   const std::vector<std::string> &headers) const;
};