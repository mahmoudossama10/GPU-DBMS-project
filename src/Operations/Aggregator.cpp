#include "../../include/Operations/Aggregator.hpp"
#include "Utilities/ErrorHandling.hpp"
#include <algorithm>
#include <unordered_set>

AggregatorPlan::AggregatorPlan(std::shared_ptr<Table> input, const std::vector<hsql::Expr *> &select_list)
    : input_(input), select_list_(select_list) {}

std::shared_ptr<Table> AggregatorPlan::execute()
{
    if (!input_ || input_->getData().empty())
    {
        return input_;
    }

    auto aggregates = parseAggregates(select_list_, *input_);
    return aggregateTable(*input_, aggregates);
}

std::vector<AggregatorPlan::AggregateOp> AggregatorPlan::parseAggregates(
    const std::vector<hsql::Expr *> &select_list, const Table &table)
{
    std::vector<AggregateOp> aggregates;

    for (const auto *expr : select_list)
    {
        if (expr->type == hsql::kExprFunctionRef && expr->name)
        {
            std::string func_name = expr->name;
            std::transform(func_name.begin(), func_name.end(), func_name.begin(), ::tolower);

            if (func_name == "count" || func_name == "sum" || func_name == "avg" ||
                func_name == "min" || func_name == "max")
            {
                AggregateOp op;
                op.function_name = func_name;
                op.is_distinct = expr->distinct;

                if (expr->exprList && !expr->exprList->empty())
                {
                    const auto *arg = expr->exprList->at(0);
                    if (arg->type == hsql::kExprColumnRef && arg->name)
                    {
                        op.column_name = arg->name;
                        if (!table.hasColumn(op.column_name))
                        {
                            throw SemanticError("Column not found for aggregate: " + op.column_name);
                        }
                    }
                    else if (arg->type == hsql::kExprStar && func_name == "count")
                    {
                        op.column_name = table.getHeaders()[0];
                    }
                    else
                    {
                        throw SemanticError("Invalid argument for aggregate function: " + func_name);
                    }
                }
                else
                {
                    throw SemanticError("No arguments provided for aggregate function: " + func_name);
                }

                op.alias = expr->alias ? expr->alias : func_name + "(" + op.column_name + ")";
                aggregates.push_back(op);
            }
            else
            {
                throw SemanticError("Unsupported aggregate function: " + func_name);
            }
        }
    }

    return aggregates;
}

std::string AggregatorPlan::unionValueToString(const unionV &value, ColumnType type)
{
    switch (type)
    {
    case ColumnType::STRING:
        return *(value.s);
    case ColumnType::INTEGER:
        return std::to_string(value.i);
    case ColumnType::DOUBLE:
        return std::to_string(value.d);
    case ColumnType::DATETIME:
    {
        char buffer[64];
        snprintf(buffer, sizeof(buffer), "%04hu-%02hu-%02hu %02hhu:%02hhu:%02hhu",
                 value.t->year, value.t->month, value.t->day,
                 value.t->hour, value.t->minute, value.t->second);
        return std::string(buffer);
    }
    default:
        throw std::runtime_error("Unknown column type");
    }
}

std::shared_ptr<Table> AggregatorPlan::aggregateTable(
    const Table &table, const std::vector<AggregateOp> &aggregates)
{
    if (aggregates.empty())
    {
        throw SemanticError("No aggregate operations to perform");
    }

    std::unordered_map<std::string, std::vector<unionV>> result_data;
    std::vector<std::string> result_headers;
    std::unordered_map<std::string, ColumnType> result_types;
    size_t num_rows = table.getSize();

    for (const auto &op : aggregates)
    {
        result_headers.push_back(op.alias);
        ColumnType col_type = table.getColumnType(op.column_name);

        if (op.function_name == "count"){
            result_types[op.alias] = ColumnType::INTEGER;
        }
        else if (op.function_name == "avg") {

            result_types[op.alias] = ColumnType::DOUBLE;
        }
        else {
            result_types[op.alias] = col_type;
        }

        std::vector<unionV> result_col(1);
        if (op.function_name == "count")
        {
            if (op.is_distinct)
            {
                std::unordered_set<std::string> unique_values;
                for (size_t i = 0; i < num_rows; ++i)
                {
                    std::string val_str;
                    switch (col_type)
                    {
                    case ColumnType::STRING:
                        val_str = table.getString(op.column_name, i);
                        break;
                    case ColumnType::INTEGER:
                        val_str = std::to_string(table.getInteger(op.column_name, i));
                        break;
                    case ColumnType::DOUBLE:
                        val_str = std::to_string(table.getDouble(op.column_name, i));
                        break;
                    case ColumnType::DATETIME:
                        val_str = unionValueToString(table.getRow(i)[table.getColumnIndex(op.column_name)], col_type);
                        break;
                    }
                    unique_values.insert(val_str);
                }
                result_col[0].i = unique_values.size();
            }
            else
            {
                result_col[0].i = num_rows;
            }
        }
        else if (op.function_name == "sum" || op.function_name == "avg")
        {
            double total = 0.0;
            size_t count = 0;
            for (size_t i = 0; i < num_rows; ++i)
            {
                if (col_type == ColumnType::INTEGER)
                {
                    total += table.getInteger(op.column_name, i);
                }
                else if (col_type == ColumnType::DOUBLE)
                {
                    total += table.getDouble(op.column_name, i);
                }
                count++;
            }
            if (op.function_name == "avg" && count > 0)
            {
                total /= count;
            }
            if (col_type == ColumnType::INTEGER && op.function_name != "avg")
            {
                result_col[0].i = static_cast<int64_t>(total);
            }
            else
            {
                result_col[0].d = total;
            }
        }
        else if (op.function_name == "min" || op.function_name == "max")
        {
            if (num_rows == 0)
            {
                throw SemanticError("No rows to compute MIN/MAX for column: " + op.column_name);
            }
            if (col_type == ColumnType::STRING)
            {
                std::string extreme = table.getString(op.column_name, 0);
                for (size_t i = 1; i < num_rows; ++i)
                {
                    std::string val = table.getString(op.column_name, i);
                    if (op.function_name == "min" ? val < extreme : val > extreme)
                    {
                        extreme = val;
                    }
                }
                result_col[0].s = new std::string(extreme);
            }
            else if (col_type == ColumnType::INTEGER)
            {
                int64_t extreme = table.getInteger(op.column_name, 0);
                for (size_t i = 1; i < num_rows; ++i)
                {
                    int64_t val = table.getInteger(op.column_name, i);
                    if (op.function_name == "min" ? val < extreme : val > extreme)
                    {
                        extreme = val;
                    }
                }
                result_col[0].i = extreme;
            }
            else if (col_type == ColumnType::DOUBLE)
            {
                double extreme = table.getDouble(op.column_name, 0);
                for (size_t i = 1; i < num_rows; ++i)
                {
                    double val = table.getDouble(op.column_name, i);
                    if (op.function_name == "min" ? val < extreme : val > extreme)
                    {
                        extreme = val;
                    }
                }
                result_col[0].d = extreme;
            }
            else if (col_type == ColumnType::DATETIME)
            {
                const dateTime &first = table.getDateTime(op.column_name, 0);
                dateTime extreme = first;
                for (size_t i = 1; i < num_rows; ++i)
                {
                    const dateTime &val = table.getDateTime(op.column_name, i);
                    bool compare;
                    if (op.function_name == "min")
                    {
                        compare = (val.year < extreme.year) ||
                                  (val.year == extreme.year && val.month < extreme.month) ||
                                  (val.year == extreme.year && val.month == extreme.month && val.day < extreme.day) ||
                                  (val.year == extreme.year && val.month == extreme.month && val.day == extreme.day && val.hour < extreme.hour) ||
                                  (val.year == extreme.year && val.month == extreme.month && val.day == extreme.day && val.hour == extreme.hour && val.minute < extreme.minute) ||
                                  (val.year == extreme.year && val.month == extreme.month && val.day == extreme.day && val.hour == extreme.hour && val.minute == extreme.minute && val.second < extreme.second);
                    }
                    else
                    {
                        compare = (val.year > extreme.year) ||
                                  (val.year == extreme.year && val.month > extreme.month) ||
                                  (val.year == extreme.year && val.month == extreme.month && val.day > extreme.day) ||
                                  (val.year == extreme.year && val.month == extreme.month && val.day == extreme.day && val.hour > extreme.hour) ||
                                  (val.year == extreme.year && val.month == extreme.month && val.day == extreme.day && val.hour == extreme.hour && val.minute > extreme.minute) ||
                                  (val.year == extreme.year && val.month == extreme.month && val.day == extreme.day && val.hour == extreme.hour && val.minute == extreme.minute && val.second > extreme.second);
                    }
                    if (compare)
                    {
                        extreme = val;
                    }
                }
                result_col[0].t = new dateTime(extreme);
            }
        }
        result_data[op.alias] = std::move(result_col);
    }

    return std::make_shared<Table>(
        table.getName() + "_aggregated",
        result_headers,
        std::move(result_data),
        result_types);
}