#include "../../include/Operations/Project.hpp"
#include "../../include/DataHandling/Table.hpp"
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

    bool isSelectAll(std::vector<hsql::Expr *> *selectList)
    {
        if (!selectList || selectList->empty())
        {
            return false;
        }

        for (auto *expr : *selectList)
        {
            if (expr->type == hsql::kExprStar)
            {
                return true;
            }
        }

        // Check for "SELECT table.*" case
        for (auto *expr : *selectList)
        {
            if (expr->type == hsql::kExprColumnRef &&
                expr->table != nullptr &&
                expr->name != nullptr &&
                strcmp(expr->name, "*") == 0)
            {
                return true;
            }
        }
        return false;
    }

    bool selectListNeedsProjection(std::vector<hsql::Expr *> &selectList)
    {
        // If * present, never project
        for (auto *expr : selectList)
            if (expr->type == hsql::kExprStar)
                return false;

        // If *table present, also never project
        for (auto *expr : selectList)
        {
            if (expr->type == hsql::kExprColumnRef && expr->name && std::strcmp(expr->name, "*") == 0)
                return false;
        }

        // If any non-aggregate expression, projection is required
        for (auto *expr : selectList)
            if (expr->type != hsql::kExprFunctionRef)
                return true;

        // All are aggregates: NO projection needed
        return false;
    }

}
ProjectPlan::ProjectPlan(std::shared_ptr<Table> input,
                         const std::vector<hsql::Expr *> &select_list)
    : input_(input), select_list_(select_list) {}

std::shared_ptr<Table> ProjectPlan::execute()
{
    auto input_table = input_;

    if (!isSelectAll(&select_list_) && selectListNeedsProjection(select_list_))
    {
        return processProjection(input_table);
    }
    return input_table;
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
            if (expr->table)
            {
                names.push_back(std::string(expr->table) + "." + expr->name);
            }
            else
            {
                names.push_back(expr->name);
            }
        }
        else
        {
            throw SemanticError("Unnamed projection expressions require aliases");
        }
    }
    return names;
}
std::string ProjectPlan::getColumnNameFromExpr(const hsql::Expr *expr) const
{
    if (expr->type == hsql::kExprColumnRef)
    {
        if (expr->table && expr->name)
        {
            // Handle table.column notation
            return std::string(expr->table) + "." + std::string(expr->name);
        }
        else
        {
            // Just column name
            return expr->name;
        }
    }
    return "";
}

std::shared_ptr<Table> ProjectPlan::processProjection(std::shared_ptr<Table> input) const
{
    // Get the input data, header names, and column types
    const auto &inputData = input->getData();
    const auto &inputColumnTypes = input->getColumnTypes();

    // Initialize maps for the new table
    std::unordered_map<std::string, std::vector<unionV>> outputData;
    std::unordered_map<std::string, ColumnType> outputColumnTypes;
    std::vector<std::string> outputHeaders = getColumnNames();

    // For each column in the select list
    for (size_t i = 0; i < select_list_.size(); i++)
    {
        const auto *expr = select_list_[i];
        std::string outputColName = outputHeaders[i]; // Use the already processed column name/alias

        if (expr->type == hsql::kExprColumnRef)
        {
            // Get the source column name
            std::string inputColName = getColumnNameFromExpr(expr);

            // Try to find the column in the input data
            auto it = inputData.find(inputColName);
            if (it == inputData.end())
            {
                // Try without table prefix
                inputColName = expr->name;
                it = inputData.find(inputColName);

                if (it == inputData.end())
                {
                    throw std::runtime_error("Column not found: " + inputColName);
                }
            }

            // Copy the column data and type to the output
            outputData[outputColName] = it->second;
            outputColumnTypes[outputColName] = inputColumnTypes.at(inputColName);
        }
        else
        {
            // For now, we only support direct column references
            throw SemanticError("Only column references are currently supported in projection");
        }
    }

    // Create and return the new table
    auto output = std::make_shared<Table>(
        input->getName() + "_projected",
        outputHeaders,
        outputData,
        outputColumnTypes);

    return output;
}
