#pragma once
#include "../DataHandling/Table.hpp"
#include "../QueryProcessing/PlanBuilder.hpp"
#include <hsql/SQLParser.h>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>

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

    // Helper to get the column name from an expression
    std::string getColumnNameFromExpr(const hsql::Expr *expr) const;
};