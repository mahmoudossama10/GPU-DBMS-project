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
