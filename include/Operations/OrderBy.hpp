#pragma once
#include "../DataHandling/Table.hpp"
#include "../QueryProcessing/PlanBuilder.hpp"
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

class OrderByPlan : public ExecutionPlan
{
public:
    OrderByPlan(std::shared_ptr<Table> input,
                const std::vector<hsql::OrderDescription *> &order_exprs);

    std::shared_ptr<Table> execute() override;

private:
    std::shared_ptr<Table> input_;
    std::vector<hsql::OrderDescription *> order_exprs_;

    struct SortColumn
    {
        size_t column_index;
        bool is_ascending;
        ColumnType type;
    };

    std::vector<SortColumn> parseOrderBy(const Table &table) const;

    static bool compareRows(const std::vector<unionV> &a,
                            const std::vector<unionV> &b,
                            const std::vector<SortColumn> &sort_cols);
};