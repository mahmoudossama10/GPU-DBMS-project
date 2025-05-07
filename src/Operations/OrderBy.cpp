#include "../../include/Operations/OrderBy.hpp"
#include "../../include/Utilities/StringUtils.hpp"
#include "Utilities/ErrorHandling.hpp"
#include <algorithm>

OrderByPlan::OrderByPlan(std::shared_ptr<Table> input,
                         const std::vector<hsql::OrderDescription *> &order_exprs)
    : input_(std::move(input)), order_exprs_(order_exprs) {}

std::shared_ptr<Table> OrderByPlan::execute()
{
    std::shared_ptr<Table> table = input_;
    if (!table || table->getData().empty())
        return table;

    std::vector<SortColumn> sort_cols = parseOrderBy(*table);

    std::vector<std::vector<unionV>> sorted_rows;
    const auto &data_map = table->getData();
    size_t num_rows = data_map.begin()->second.size();
    sorted_rows.resize(num_rows);

    for (const auto &[col_name, col_values] : data_map)
    {
        for (size_t i = 0; i < num_rows; ++i)
        {
            if (sorted_rows[i].size() < data_map.size())
                sorted_rows[i].resize(data_map.size());
            sorted_rows[i][table->getColumnIndex(col_name)] = col_values[i];
        }
    }

    std::sort(sorted_rows.begin(), sorted_rows.end(),
              [&](const std::vector<unionV> &a, const std::vector<unionV> &b)
              {
                  return compareRows(a, b, sort_cols);
              });

    std::unordered_map<std::string, std::vector<unionV>> sorted_data_map;
    const auto &headers = table->getHeaders();
    for (size_t col_idx = 0; col_idx < headers.size(); ++col_idx)
    {
        std::vector<unionV> column_data;
        column_data.reserve(sorted_rows.size());
        for (const auto &row : sorted_rows)
        {
            column_data.push_back(row[col_idx]);
        }
        sorted_data_map[headers[col_idx]] = std::move(column_data);
    }

    return std::make_shared<Table>(
        table->getName() + "_ordered",
        table->getHeaders(),
        std::move(sorted_data_map),
        table->getColumnTypes());
}

std::vector<OrderByPlan::SortColumn> OrderByPlan::parseOrderBy(const Table &table) const
{
    std::vector<SortColumn> sort_cols;
    const auto &headers = table.getHeaders();

    for (const auto *order_desc : order_exprs_)
    {
        if (order_desc->type != hsql::kOrderAsc && order_desc->type != hsql::kOrderDesc)
            throw std::runtime_error("Unsupported ORDER BY type");

        const hsql::Expr *expr = order_desc->expr;
        if (!expr || expr->type != hsql::kExprColumnRef)
            throw std::runtime_error("Only column references are supported in ORDER BY");

        // Extract column name with optional table alias (e.g., "a.age" -> "a.age")
        std::string col_name;
        if (expr->table != nullptr)
        {
            col_name = std::string(expr->table) + "." + expr->name;
        }
        else
        {
            col_name = expr->name;
        }
        if (!table.hasColumn(col_name))
        {
            col_name = expr->name;
        }
        // Find the column index
        size_t col_idx = table.getColumnIndex(col_name);

        // Fallback: If not found, search by column name only (without alias)
        if (col_idx >= headers.size() || headers[col_idx] != col_name)
        {
            for (size_t i = 0; i < headers.size(); ++i)
            {
                if (headers[i] == expr->name)
                { // Check without alias
                    col_idx = i;
                    break;
                }
            }
        }

        // Validate column existence
        if (col_idx >= headers.size() || headers[col_idx] != col_name)
        {
            throw std::runtime_error("Column '" + col_name + "' not found in table for ORDER BY");
        }

        ColumnType col_type = table.getColumnType(col_name);
        sort_cols.push_back({col_idx, order_desc->type == hsql::kOrderAsc, col_type});
    }

    return sort_cols;
}

bool OrderByPlan::compareRows(const std::vector<unionV> &a,
                              const std::vector<unionV> &b,
                              const std::vector<SortColumn> &sort_cols)
{
    for (const auto &sort_col : sort_cols)
    {
        const unionV &val_a = a[sort_col.column_index];
        const unionV &val_b = b[sort_col.column_index];

        int cmp = 0;
        switch (sort_col.type)
        {
        case ColumnType::INTEGER:
            cmp = (val_a.i < val_b.i) ? -1 : (val_a.i > val_b.i ? 1 : 0);
            break;
        case ColumnType::DOUBLE:
            cmp = (val_a.d < val_b.d) ? -1 : (val_a.d > val_b.d ? 1 : 0);
            break;
        case ColumnType::STRING:
            cmp = val_a.s->compare(*val_b.s);
            break;
        case ColumnType::DATETIME:
            cmp = (val_a.d < val_b.d) ? -1 : (val_a.d > val_b.d ? 1 : 0);
            break;
        default:
            throw SemanticError("Unsupported column type in ORDER BY");
        }

        if (cmp != 0)
            return sort_col.is_ascending ? cmp < 0 : cmp > 0;
    }
    return false;
}