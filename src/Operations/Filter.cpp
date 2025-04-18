#include "../../include/Operations/Filter.hpp"
#include "Utilities/ErrorHandling.hpp"
#include <algorithm>
#include <unordered_map>
#include <cctype>
#include <sstream>
#include <iostream>
#include <algorithm> // for to_lower
#include <cmath>     // for NaN
#include <regex>
#include <hsql/util/sqlhelper.h>

namespace
{

    void printExpr(const hsql::Expr *expr)
    {
        if (expr == nullptr)
        {
            std::cout << "Expression is null\n";
            return;
        }

        switch (expr->type)
        {
        case hsql::kExprOperator:
        {
            std::cout << "Operator Expression: ";

            if (expr->expr)
            {
                printExpr(expr->expr); // left
            }
            if (expr->expr2)
            {
                std::cout << " ";
                printExpr(expr->expr2); // right
            }
            break;
        }

        case hsql::kExprColumnRef:
            std::cout << "Column: " << expr->name << "\n";
            break;

        case hsql::kExprLiteralString:
            std::cout << "String Literal: \"" << expr->name << "\"\n";
            break;

        case hsql::kExprLiteralInt:
            std::cout << "Int Literal: " << expr->ival << "\n";
            break;

        case hsql::kExprLiteralFloat:
            std::cout << "Float Literal: " << expr->fval << "\n";
            break;

        default:
            std::cout << "Unhandled expression type: " << expr->type << "\n";
            break;
        }
    }

    std::unordered_map<std::string, size_t> createColumnIndexMap(
        const std::vector<std::string> &headers)
    {
        std::unordered_map<std::string, size_t> indexMap;
        for (size_t i = 0; i < headers.size(); ++i)
        {
            indexMap[headers[i]] = i;
        }
        return indexMap;
    }

    bool tryConvertToNumber(const std::string &str, double &num)
    {
        std::istringstream iss(str);
        return (iss >> num) && iss.eof();
    }
}

std::shared_ptr<Table> FilterPlan::execute()
{

    auto table = input_->execute();
    return Filter::apply(table, condition_);
}

std::shared_ptr<Table> Filter::apply(
    std::shared_ptr<Table> table,
    const hsql::Expr *condition)
{
    // hsql::printExpression(const_cast<hsql::Expr *>(condition), 5);
    if (!condition)
    {
        throw SemanticError("Null condition in filter");
    }

    const auto &headers = table->getHeaders();
    const auto &data = table->getData();

    // Create column index map once for efficiency
    const auto columnIndexMap = createColumnIndexMap(headers);

    std::vector<std::vector<std::string>> filteredData;
    filteredData.reserve(data.size());

    std::copy_if(
        data.begin(), data.end(),
        std::back_inserter(filteredData),
        [&](const std::vector<std::string> &row)
        {
            // Pass all required arguments including columnIndexMap
            return evaluateCondition(row, condition, headers, columnIndexMap);
        });

    return std::make_shared<Table>(
        table->getName() + "_filtered",
        headers,
        filteredData);
}

bool Filter::evaluateCondition(
    const std::vector<std::string> &row,
    const hsql::Expr *condition,
    const std::vector<std::string> &headers,
    const std::unordered_map<std::string, size_t> &columnIndexMap) // Pass map by reference
{
    if (!condition)
        return true;

    switch (condition->type)
    {
    case hsql::kExprOperator:
        return handleOperator(row, condition, headers, columnIndexMap);

    case hsql::kExprLiteralNull:
        return handleNullCondition(row, condition, columnIndexMap);

    default:
        throw SemanticError("Unsupported condition type: " +
                            std::to_string(condition->type));
    }
}

bool Filter::handleOperator(
    const std::vector<std::string> &row,
    const hsql::Expr *expr,
    const std::vector<std::string> &headers,
    const std::unordered_map<std::string, size_t> &columnIndexMap)
{
    switch (expr->opType)
    {
    // Logical operators
    case hsql::kOpAnd:
        return evaluateCondition(row, expr->expr, headers, columnIndexMap) &&
               evaluateCondition(row, expr->expr2, headers, columnIndexMap);

    case hsql::kOpOr:
        return evaluateCondition(row, expr->expr, headers, columnIndexMap) ||
               evaluateCondition(row, expr->expr2, headers, columnIndexMap);

    case hsql::kOpNot:
        return !evaluateCondition(row, expr->expr, headers, columnIndexMap);

    // Comparison operators
    case hsql::kOpEquals:
    case hsql::kOpNotEquals:
    case hsql::kOpLess:
    case hsql::kOpLessEq:
    case hsql::kOpGreater:
    case hsql::kOpGreaterEq:
    case hsql::kOpLike:
    case hsql::kOpNotLike:
        return handleComparison(row, expr, headers, columnIndexMap);

    default:
        throw SemanticError("Unsupported operator: " +
                            std::to_string(expr->opType));
    }
}
bool Filter::handleComparison(
    const std::vector<std::string> &row,
    const hsql::Expr *expr,
    const std::vector<std::string> &headers,
    const std::unordered_map<std::string, size_t> &columnIndexMap)
{
    auto getValue = [&](const hsql::Expr *e) -> std::string
    {
        if (e->type == hsql::kExprColumnRef)
        {
            std::string full_column_name;

            // Handle aliased columns (table.column)
            if (e->table != nullptr && e->table[0] != '\0')
            {
                full_column_name = std::string(e->table) + "." + e->name;
            }
            // Handle unaliased columns
            else
            {
                full_column_name = e->name;
            }

            try
            {
                return row[columnIndexMap.at(full_column_name)];
            }
            catch (const std::out_of_range &)
            {
                // Fallback to try without table prefix
                try
                {
                    return row[columnIndexMap.at(e->name)];
                }
                catch (const std::out_of_range &)
                {
                    throw SemanticError("Column not found: " + full_column_name);
                }
            }
        }
        else if (e->type == hsql::kExprLiteralInt)
        {
            return std::to_string(e->ival);
        }
        else if (e->type == hsql::kExprLiteralFloat)
        {
            return std::to_string(e->fval);
        }
        else if (e->type == hsql::kExprLiteralString)
        {
            return e->name;
        }
        throw SemanticError("Unsupported comparison operand");
    };

    const std::string lhs = getValue(expr->expr);
    const std::string rhs = getValue(expr->expr2);

    // Numeric comparison if possible
    try
    {
        double numLhs = std::stod(lhs);
        double numRhs = std::stod(rhs);
        return compareNumerics(numLhs, numRhs, expr->opType);
    }
    catch (...)
    {
        // Fall back to string comparison
        return compareStrings(lhs, rhs, expr->opType);
    }
}

bool Filter::compareNumerics(double lhs, double rhs, hsql::OperatorType op)
{
    switch (op)
    {
    case hsql::kOpEquals:
        return lhs == rhs;
    case hsql::kOpNotEquals:
        return lhs != rhs;
    case hsql::kOpLess:
        return lhs < rhs;
    case hsql::kOpLessEq:
        return lhs <= rhs;
    case hsql::kOpGreater:
        return lhs > rhs;
    case hsql::kOpGreaterEq:
        return lhs >= rhs;
    default:
        throw SemanticError("Unsupported numeric operator");
    }
}

bool Filter::compareStrings(const std::string &lhs, const std::string &rhs, hsql::OperatorType op)
{
    switch (op)
    {
    case hsql::kOpEquals:
        return lhs == rhs;
    case hsql::kOpNotEquals:
        return lhs != rhs;
    case hsql::kOpLess:
        return lhs < rhs;
    case hsql::kOpLessEq:
        return lhs <= rhs;
    case hsql::kOpGreater:
        return lhs > rhs;
    case hsql::kOpGreaterEq:
        return lhs >= rhs;
    case hsql::kOpLike:
        return matchLikePattern(lhs, rhs);
    case hsql::kOpNotLike:
        return !matchLikePattern(lhs, rhs);
    default:
        throw SemanticError("Unsupported string operator");
    }
}

bool Filter::handleNullCondition(
    const std::vector<std::string> &row,
    const hsql::Expr *expr,
    const std::unordered_map<std::string, size_t> &columnIndexMap)
{
    const std::string &column = expr->expr->name;
    const std::string &value = row[columnIndexMap.at(column)];
    return value.empty() || value == "NULL";
}

bool Filter::matchLikePattern(const std::string &value, const std::string &pattern)
{
    // Implement SQL LIKE pattern matching
    std::string regexPattern;
    regexPattern.reserve(pattern.size() * 2);

    // Convert SQL LIKE pattern to regex
    for (char c : pattern)
    {
        switch (c)
        {
        case '%':
            regexPattern += ".*";
            break;
        case '_':
            regexPattern += '.';
            break;
        case '.':
            regexPattern += "\\.";
            break;
        default:
            regexPattern += c;
            break;
        }
    }

    std::regex re(regexPattern, std::regex_constants::icase);
    return std::regex_match(value, re);
}

bool Filter::compareValues(
    const std::string &lhs,
    const std::string &rhs,
    hsql::OperatorType op)
{
    // Try numeric comparison first

    double lhs_num, rhs_num;
    bool numeric_comparison = tryConvertToNumber(lhs, lhs_num) &&
                              tryConvertToNumber(rhs, rhs_num);

    if (numeric_comparison)
    {
        switch (op)
        {
        case hsql::kOpEquals:
            return lhs_num == rhs_num;
        case hsql::kOpNotEquals:
            return lhs_num != rhs_num;
        case hsql::kOpLess:
            return lhs_num < rhs_num;
        case hsql::kOpLessEq:
            return lhs_num <= rhs_num;
        case hsql::kOpGreater:
            return lhs_num > rhs_num;
        case hsql::kOpGreaterEq:
            return lhs_num >= rhs_num;
        default:
            throw SemanticError("Unsupported comparison operator");
        }
    }

    // Fall back to string comparison
    switch (op)
    {
    case hsql::kOpEquals:
        return lhs == rhs;
    case hsql::kOpNotEquals:
        return lhs != rhs;
    case hsql::kOpLess:
        return lhs < rhs;
    case hsql::kOpLessEq:
        return lhs <= rhs;
    case hsql::kOpGreater:
        return lhs > rhs;
    case hsql::kOpGreaterEq:
        return lhs >= rhs;
    case hsql::kOpLike:
        return handleLikeOperator(lhs, rhs);
    case hsql::kOpNotLike:
        return !handleLikeOperator(lhs, rhs);
    default:
        throw SemanticError("Unsupported comparison operator");
    }
}

bool Filter::handleLikeOperator(const std::string &value, const std::string &pattern)
{
    // Basic LIKE operator implementation (supports % and _ wildcards)
    size_t v_pos = 0;
    size_t p_pos = 0;
    size_t v_len = value.length();
    size_t p_len = pattern.length();

    while (p_pos < p_len)
    {
        if (pattern[p_pos] == '%')
        {
            // % matches 0 or more characters
            p_pos++;
            if (p_pos == p_len)
                return true; // % at end matches all

            // Look ahead for next character in pattern
            while (v_pos < v_len)
            {
                if ((pattern[p_pos] == '_' || pattern[p_pos] == value[v_pos]) &&
                    handleLikeOperator(value.substr(v_pos), pattern.substr(p_pos)))
                {
                    return true;
                }
                v_pos++;
            }
            return false;
        }
        else if (v_pos >= v_len ||
                 (pattern[p_pos] != '_' && pattern[p_pos] != value[v_pos]))
        {
            return false;
        }
        p_pos++;
        v_pos++;
    }

    return v_pos == v_len;
}