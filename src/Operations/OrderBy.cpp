#include "../../include/Operations/OrderBy.hpp"
#include "../../include/Utilities/ErrorHandling.hpp"
#include "../../include/Utilities/StringUtils.hpp"
#include "../../include/Operations/GPUAggregator.hpp"
#include <algorithm>
#include <stdexcept>
#include <cmath>

using namespace StringUtils;

OrderByPlan::OrderByPlan(std::unique_ptr<ExecutionPlan> input,
                         const std::vector<hsql::OrderDescription *> &order_exprs)
    : input_(std::move(input)), order_exprs_(order_exprs) {}

std::shared_ptr<Table> OrderByPlan::execute()
{
    auto input_table = input_->execute();
    auto sort_cols = parseOrderBy(*input_table);

    // --- Only support single int ORDER BY in this version ---
    if (sort_cols.size() != 1)
        throw SemanticError("Only single-column ORDER BY supported for GPU.");

    const auto& colname = input_table->getHeaders()[sort_cols[0].column_index];
    bool ascending = sort_cols[0].is_ascending;

    // Try int column (fastest, canonical case for benchmarking!)
    try {
        // Use your Table column cache
        const auto& col_data = input_table->getIntColumn(colname);

        // GPU sort
        std::vector<int> indices = GPUAggregator::gpuArgsortInt(col_data, ascending);

        // Reorder rows
        std::vector<std::vector<std::string>> sorted_data;
        sorted_data.reserve(indices.size());
        for (int idx : indices)
            sorted_data.push_back(input_table->getRow(idx));

        // Return sorted table
        return std::make_shared<Table>(
            input_table->getName() + "_sorted",
            input_table->getHeaders(),
            sorted_data);
    } catch (...) {
        // Fallback to CPU or string-based sort if int conversion fails
        auto sorted_data = input_table->getData();
        std::sort(sorted_data.begin(), sorted_data.end(),
            [&](const auto& a, const auto& b) {
                return ascending
                    ? a[sort_cols[0].column_index] < b[sort_cols[0].column_index]
                    : a[sort_cols[0].column_index] > b[sort_cols[0].column_index];
            });
        return std::make_shared<Table>(
            input_table->getName() + "_sorted",
            input_table->getHeaders(),
            sorted_data);
    }
}

std::vector<OrderByPlan::SortColumn> OrderByPlan::parseOrderBy(const Table &table) const
{
    std::vector<SortColumn> sort_cols;

    for (const auto *order_desc : order_exprs_)
    {
        const auto *expr = order_desc->expr;
        if (expr->type != hsql::kExprColumnRef)
        {
            throw SemanticError("Complex ORDER BY expressions not supported");
        }

        SortColumn sc;
        sc.column_index = table.getColumnIndex(expr->name);
        sc.is_ascending = (order_desc->type == hsql::kOrderAsc);

        // Detect numeric column
        sc.is_numeric = true;
        try
        {
            for (const auto &row : table.getData())
            {
                std::stod(row[sc.column_index]);
            }
        }
        catch (...)
        {
            sc.is_numeric = false;
        }

        sort_cols.push_back(sc);
    }
    return sort_cols;
}

bool OrderByPlan::compareRows(const std::vector<std::string> &a,
                              const std::vector<std::string> &b,
                              const std::vector<SortColumn> &sort_cols)
{
    for (const auto &sc : sort_cols)
    {
        const auto &val_a = a[sc.column_index];
        const auto &val_b = b[sc.column_index];

        // Handle NULLs as smallest possible values
        if (val_a.empty() != val_b.empty())
        {
            return sc.is_ascending ? val_a.empty() : val_b.empty();
        }

        int comparison = 0;
        if (sc.is_numeric)
        {
            double num_a = std::stod(val_a);
            double num_b = std::stod(val_b);
            comparison = (num_a < num_b) ? -1 : (num_a > num_b) ? 1
                                                                : 0;
        }
        else
        {
            comparison = toLower(val_a).compare(toLower(val_b));
        }

        if (comparison != 0)
        {
            return sc.is_ascending ? (comparison < 0) : (comparison > 0);
        }
    }
    return false; // Equal for all sort columns
}