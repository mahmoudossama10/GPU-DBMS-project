#include "../../include/Operations/Project.hpp"
#include "Utilities/ErrorHandling.hpp"
#include <algorithm>
#include <unordered_map>
#include <cctype>
#include <sstream>
#include <iostream>
#include <algorithm> // for to_lower
#include <cmath>     // for NaN
#include <regex>

namespace
{

    static size_t getColumnIndex(const std::string &column_name,
                                 const std::vector<std::string> &headers)
    {
        auto it = std::find_if(headers.begin(), headers.end(),
                               [&column_name](const std::string &header)
                               {
                                   return std::equal(header.begin(), header.end(),
                                                     column_name.begin(), column_name.end(),
                                                     [](char a, char b)
                                                     {
                                                         return std::tolower(a) == std::tolower(b);
                                                     });
                               });

        if (it != headers.end())
        {
            return std::distance(headers.begin(), it);
        }
        throw std::runtime_error("Column '" + column_name + "' not found in table");
    }
}
ProjectPlan::ProjectPlan(std::unique_ptr<ExecutionPlan> input,
                         const std::vector<hsql::Expr *> &select_list)
    : input_(std::move(input)), select_list_(select_list) {}

std::shared_ptr<Table> ProjectPlan::execute()
{
    auto input_table = input_->execute();
    return processProjection(input_table);
}

std::vector<std::string> ProjectPlan::getColumnNames() const
{
    std::vector<std::string> names;
    for (const auto *expr : select_list_)
    {
        if (expr->alias)
        {
            names.push_back(expr->alias);
        }
        else if (expr->type == hsql::kExprColumnRef)
        {
            names.push_back(expr->name);
        }
        else
        {
            std::cout<<expr->type<<'\n';
            throw SemanticError("Unnamed projection expressions require aliases");
        }
    }
    return names;
}

std::string ProjectPlan::evaluateExpression(const std::vector<std::string> &row,
                                            const hsql::Expr *expr,
                                            const std::vector<std::string> &headers) const
{
    // Reuse Filter's column reference handling
    if (expr->type == hsql::kExprColumnRef)
    {
        size_t col_idx = getColumnIndex(expr->name, headers);
        return row[col_idx];
    }
    // Handle literals
    else if (expr->type == hsql::kExprLiteralInt)
    {
        return std::to_string(expr->ival);
    }
    else if (expr->type == hsql::kExprLiteralFloat)
    {
        return std::to_string(expr->fval);
    }
    else if (expr->type == hsql::kExprLiteralString)
    {
        return expr->name;
    }
    // Handle simple arithmetic operations
    else if (expr->type == hsql::kExprOperator)
    {
        std::string lhs = evaluateExpression(row, expr->expr, headers);
        std::string rhs = evaluateExpression(row, expr->expr2, headers);

        switch (expr->opType)
        {
        case hsql::kOpPlus:
            return std::to_string(std::stod(lhs) + std::stod(rhs));
        case hsql::kOpMinus:
            return std::to_string(std::stod(lhs) - std::stod(rhs));
        case hsql::kOpAsterisk:
            return std::to_string(std::stod(lhs) * std::stod(rhs));
        case hsql::kOpSlash:
            return std::to_string(std::stod(lhs) / std::stod(rhs));
        default:
            throw SemanticError("Unsupported operator in projection");
        }
    }
    throw SemanticError("Unsupported expression type in projection");
}

std::shared_ptr<Table> ProjectPlan::processProjection(std::shared_ptr<Table> input) const
{
    std::vector<std::vector<std::string>> dataRows;
    auto output = std::make_shared<Table>(
        input->getName() + "_projected",
        getColumnNames(),
        dataRows);

    const auto &headers = input->getHeaders();
    for (const auto &row : input->getData())
    {
        std::vector<std::string> projected_row;
        for (const auto *expr : select_list_)
        {
            projected_row.push_back(evaluateExpression(row, expr, headers));
        }
        output->addRow(projected_row);
    }

    return output;
}