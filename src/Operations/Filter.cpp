#include "../../include/Operations/Filter.hpp"
#include "Utilities/ErrorHandling.hpp"
#include <algorithm>
#include <cctype>
#include <sstream>
#include <iostream>
#include <cmath> // for NaN
#include <regex>
#include <hsql/util/sqlhelper.h>
#include <iomanip> // Required for std::setw

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
    if (!condition)
    {
        throw SemanticError("Null condition in filter");
    }

    const auto &headers = table->getHeaders();
    const auto &columnTypes = table->getColumnTypes();
    const int rowCount = table->getSize();

    // Create a new Table with the same structure
    std::unordered_map<std::string, std::vector<unionV>> filteredColumnData;
    for (const auto &header : headers)
    {
        filteredColumnData[header] = std::vector<unionV>();
    }

    // Process each row and add matching rows to the filtered data
    for (int rowIndex = 0; rowIndex < rowCount; ++rowIndex)
    {
        if (evaluateCondition(table, rowIndex, condition))
        {
            // For each matching row, add its values to the filtered data
            for (const auto &header : headers)
            {
                ColumnType type = table->getColumnType(header);
                const auto &sourceData = table->getData().at(header);
                unionV value = sourceData[rowIndex];

                // Create a deep copy of the union value if needed
                switch (type)
                {
                case ColumnType::STRING:
                {
                    std::string *newStr = new std::string(*(value.s));
                    unionV newValue;
                    newValue.s = newStr;
                    filteredColumnData[header].push_back(newValue);
                }
                break;
                case ColumnType::DATETIME:
                {
                    dateTime *newTime = new dateTime(*(value.t));
                    unionV newValue;
                    newValue.t = newTime;
                    filteredColumnData[header].push_back(newValue);
                }
                break;
                case ColumnType::INTEGER:
                case ColumnType::DOUBLE:
                    // These don't need deep copy
                    filteredColumnData[header].push_back(value);
                    break;
                }
            }
        }
    }

    return std::make_shared<Table>(
        table->getName() + "_filtered",
        headers,
        filteredColumnData,
        columnTypes);
}

bool Filter::evaluateCondition(
    const std::shared_ptr<Table> &table,
    size_t rowIndex,
    const hsql::Expr *condition)
{
    if (!condition)
        return true;

    switch (condition->type)
    {
    case hsql::kExprOperator:
        return handleOperator(table, rowIndex, condition);

    case hsql::kExprLiteralNull:
        return handleNullCondition(table, rowIndex, condition);

    default:
        throw SemanticError("Unsupported condition type: " +
                            std::to_string(condition->type));
    }
}

bool Filter::handleOperator(
    const std::shared_ptr<Table> &table,
    size_t rowIndex,
    const hsql::Expr *expr)
{
    switch (expr->opType)
    {
    // Logical operators
    case hsql::kOpAnd:
        return evaluateCondition(table, rowIndex, expr->expr) &&
               evaluateCondition(table, rowIndex, expr->expr2);

    case hsql::kOpOr:
        return evaluateCondition(table, rowIndex, expr->expr) ||
               evaluateCondition(table, rowIndex, expr->expr2);

    case hsql::kOpNot:
        return !evaluateCondition(table, rowIndex, expr->expr);

    // Comparison operators
    case hsql::kOpEquals:
    case hsql::kOpNotEquals:
    case hsql::kOpLess:
    case hsql::kOpLessEq:
    case hsql::kOpGreater:
    case hsql::kOpGreaterEq:
    case hsql::kOpLike:
    case hsql::kOpNotLike:
        return handleComparison(table, rowIndex, expr);

    default:
        throw SemanticError("Unsupported operator: " +
                            std::to_string(expr->opType));
    }
}

unionV Filter::getExprValue(
    const std::shared_ptr<Table> &table,
    size_t rowIndex,
    const hsql::Expr *expr,
    ColumnType &outType)
{
    unionV result;

    if (expr->type == hsql::kExprColumnRef)
    {
        std::string columnName;

        // Handle aliased columns (table.column)
        if (expr->table != nullptr && expr->table[0] != '\0')
        {
            columnName = std::string(expr->table) + "." + expr->name;
            if (!table->hasColumn(columnName))
            {
                // Try just column name if not found with table alias
                columnName = expr->name;
            }
        }
        else
        {
            columnName = expr->name;
        }

        if (!table->hasColumn(columnName))
        {
            throw SemanticError("Column not found: " + columnName);
        }

        outType = table->getColumnType(columnName);
        const auto &columnData = table->getData().at(columnName);
        return columnData[rowIndex];
    }
    else if (expr->type == hsql::kExprLiteralInt)
    {
        outType = ColumnType::INTEGER;
        result.i = expr->ival;
    }
    else if (expr->type == hsql::kExprLiteralFloat)
    {
        outType = ColumnType::DOUBLE;
        result.d = expr->fval;
    }
    else if (expr->type == hsql::kExprLiteralString)
    {
        outType = ColumnType::STRING;
        result.s = new std::string(expr->name);
    }
    else
    {
        throw SemanticError("Unsupported expression type in comparison");
    }

    return result;
}

std::string Filter::unionToString(const unionV &value, ColumnType type)
{
    std::stringstream ss;

    switch (type)
    {
    case ColumnType::STRING:
        return *(value.s);
    case ColumnType::INTEGER:
        ss << value.i;
        return ss.str();
    case ColumnType::DOUBLE:
        ss << value.d;
        return ss.str();
    case ColumnType::DATETIME:
    {
        const dateTime &dt = *(value.t);
        ss << dt.year << "-";
        ss << std::setw(2) << std::setfill('0') << dt.month << "-";
        ss << std::setw(2) << std::setfill('0') << dt.day << " ";
        ss << std::setw(2) << std::setfill('0') << (int)dt.hour << ":";
        ss << std::setw(2) << std::setfill('0') << (int)dt.minute << ":";
        ss << std::setw(2) << std::setfill('0') << (int)dt.second;
        return ss.str();
    }
    default:
        return "Unknown type";
    }
}

bool Filter::handleComparison(
    const std::shared_ptr<Table> &table,
    size_t rowIndex,
    const hsql::Expr *expr)
{
    ColumnType lhsType, rhsType;
    unionV lhsValue = getExprValue(table, rowIndex, expr->expr, lhsType);
    unionV rhsValue = getExprValue(table, rowIndex, expr->expr2, rhsType);

    // If types don't match, convert to the more precise type
    if (lhsType != rhsType)
    {
        // If either side is a string and operation is LIKE or NOT LIKE, convert both to strings
        if ((expr->opType == hsql::kOpLike || expr->opType == hsql::kOpNotLike))
        {
            std::string lhsStr = unionToString(lhsValue, lhsType);
            std::string rhsStr = unionToString(rhsValue, rhsType);

            // Clean up any dynamically allocated memory
            if (lhsType == ColumnType::STRING)
                delete lhsValue.s;
            if (rhsType == ColumnType::STRING)
                delete rhsValue.s;
            if (lhsType == ColumnType::DATETIME)
                delete lhsValue.t;
            if (rhsType == ColumnType::DATETIME)
                delete rhsValue.t;

            return expr->opType == hsql::kOpLike ? matchLikePattern(lhsStr, rhsStr) : !matchLikePattern(lhsStr, rhsStr);
        }

        // Handle type conversion for comparison
        if ((lhsType == ColumnType::INTEGER && rhsType == ColumnType::DOUBLE) ||
            (lhsType == ColumnType::DOUBLE && rhsType == ColumnType::INTEGER))
        {
            // Convert to double comparison
            double lhsDouble = (lhsType == ColumnType::INTEGER) ? static_cast<double>(lhsValue.i) : lhsValue.d;
            double rhsDouble = (rhsType == ColumnType::INTEGER) ? static_cast<double>(rhsValue.i) : rhsValue.d;
            return compareDoubles(lhsDouble, rhsDouble, expr->opType);
        }
        else
        {
            // Convert to string for mixed type comparisons
            std::string lhsStr = unionToString(lhsValue, lhsType);
            std::string rhsStr = unionToString(rhsValue, rhsType);

            // Clean up any dynamically allocated memory
            if (lhsType == ColumnType::STRING)
                delete lhsValue.s;
            if (rhsType == ColumnType::STRING)
                delete rhsValue.s;
            if (lhsType == ColumnType::DATETIME)
                delete lhsValue.t;
            if (rhsType == ColumnType::DATETIME)
                delete rhsValue.t;

            return compareStrings(lhsStr, rhsStr, expr->opType);
        }
    }

    // Same types, perform type-specific comparison
    bool result;
    switch (lhsType)
    {
    case ColumnType::STRING:
    {
        if (expr->opType == hsql::kOpLike || expr->opType == hsql::kOpNotLike)
        {
            result = expr->opType == hsql::kOpLike ? matchLikePattern(*(lhsValue.s), *(rhsValue.s)) : !matchLikePattern(*(lhsValue.s), *(rhsValue.s));
        }
        else
        {
            result = compareStrings(*(lhsValue.s), *(rhsValue.s), expr->opType);
        }
        delete lhsValue.s;
        delete rhsValue.s;
    }
    break;

    case ColumnType::INTEGER:
        result = compareIntegers(lhsValue.i, rhsValue.i, expr->opType);
        break;

    case ColumnType::DOUBLE:
        result = compareDoubles(lhsValue.d, rhsValue.d, expr->opType);
        break;

    case ColumnType::DATETIME:
    {
        result = compareDateTimes(*(lhsValue.t), *(rhsValue.t), expr->opType);
        delete lhsValue.t;
        delete rhsValue.t;
    }
    break;

    default:
        throw SemanticError("Unsupported data type in comparison");
    }

    return result;
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
    default:
        throw SemanticError("Unsupported string comparison operator");
    }
}

bool Filter::compareIntegers(int64_t lhs, int64_t rhs, hsql::OperatorType op)
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
        throw SemanticError("Unsupported integer comparison operator");
    }
}

bool Filter::compareDoubles(double lhs, double rhs, hsql::OperatorType op)
{
    switch (op)
    {
    case hsql::kOpEquals:
        return std::abs(lhs - rhs) < 1e-10; // Use epsilon for floating point equality
    case hsql::kOpNotEquals:
        return std::abs(lhs - rhs) >= 1e-10;
    case hsql::kOpLess:
        return lhs < rhs;
    case hsql::kOpLessEq:
        return lhs <= rhs;
    case hsql::kOpGreater:
        return lhs > rhs;
    case hsql::kOpGreaterEq:
        return lhs >= rhs;
    default:
        throw SemanticError("Unsupported double comparison operator");
    }
}

bool Filter::compareDateTimes(const dateTime &lhs, const dateTime &rhs, hsql::OperatorType op)
{
    // Convert both datetime values to comparable format (e.g., seconds since epoch)
    // This is a simplified comparison that just compares fields in order of significance
    if (lhs.year != rhs.year)
    {
        switch (op)
        {
        case hsql::kOpEquals:
            return false;
        case hsql::kOpNotEquals:
            return true;
        case hsql::kOpLess:
            return lhs.year < rhs.year;
        case hsql::kOpLessEq:
            return lhs.year < rhs.year;
        case hsql::kOpGreater:
            return lhs.year > rhs.year;
        case hsql::kOpGreaterEq:
            return lhs.year > rhs.year;
        default:
            throw SemanticError("Unsupported datetime comparison operator");
        }
    }

    if (lhs.month != rhs.month)
    {
        switch (op)
        {
        case hsql::kOpEquals:
            return false;
        case hsql::kOpNotEquals:
            return true;
        case hsql::kOpLess:
            return lhs.month < rhs.month;
        case hsql::kOpLessEq:
            return lhs.month < rhs.month;
        case hsql::kOpGreater:
            return lhs.month > rhs.month;
        case hsql::kOpGreaterEq:
            return lhs.month > rhs.month;
        default:
            throw SemanticError("Unsupported datetime comparison operator");
        }
    }

    if (lhs.day != rhs.day)
    {
        switch (op)
        {
        case hsql::kOpEquals:
            return false;
        case hsql::kOpNotEquals:
            return true;
        case hsql::kOpLess:
            return lhs.day < rhs.day;
        case hsql::kOpLessEq:
            return lhs.day < rhs.day;
        case hsql::kOpGreater:
            return lhs.day > rhs.day;
        case hsql::kOpGreaterEq:
            return lhs.day > rhs.day;
        default:
            throw SemanticError("Unsupported datetime comparison operator");
        }
    }

    if (lhs.hour != rhs.hour)
    {
        switch (op)
        {
        case hsql::kOpEquals:
            return false;
        case hsql::kOpNotEquals:
            return true;
        case hsql::kOpLess:
            return lhs.hour < rhs.hour;
        case hsql::kOpLessEq:
            return lhs.hour < rhs.hour;
        case hsql::kOpGreater:
            return lhs.hour > rhs.hour;
        case hsql::kOpGreaterEq:
            return lhs.hour > rhs.hour;
        default:
            throw SemanticError("Unsupported datetime comparison operator");
        }
    }

    if (lhs.minute != rhs.minute)
    {
        switch (op)
        {
        case hsql::kOpEquals:
            return false;
        case hsql::kOpNotEquals:
            return true;
        case hsql::kOpLess:
            return lhs.minute < rhs.minute;
        case hsql::kOpLessEq:
            return lhs.minute < rhs.minute;
        case hsql::kOpGreater:
            return lhs.minute > rhs.minute;
        case hsql::kOpGreaterEq:
            return lhs.minute > rhs.minute;
        default:
            throw SemanticError("Unsupported datetime comparison operator");
        }
    }

    if (lhs.second != rhs.second)
    {
        switch (op)
        {
        case hsql::kOpEquals:
            return false;
        case hsql::kOpNotEquals:
            return true;
        case hsql::kOpLess:
            return lhs.second < rhs.second;
        case hsql::kOpLessEq:
            return lhs.second < rhs.second;
        case hsql::kOpGreater:
            return lhs.second > rhs.second;
        case hsql::kOpGreaterEq:
            return lhs.second > rhs.second;
        default:
            throw SemanticError("Unsupported datetime comparison operator");
        }
    }

    // All fields are equal
    switch (op)
    {
    case hsql::kOpEquals:
        return true;
    case hsql::kOpNotEquals:
        return false;
    case hsql::kOpLess:
        return false;
    case hsql::kOpLessEq:
        return true;
    case hsql::kOpGreater:
        return false;
    case hsql::kOpGreaterEq:
        return true;
    default:
        throw SemanticError("Unsupported datetime comparison operator");
    }
}

bool Filter::handleNullCondition(
    const std::shared_ptr<Table> &table,
    size_t rowIndex,
    const hsql::Expr *expr)
{
    if (!expr->expr || expr->expr->type != hsql::kExprColumnRef)
    {
        throw SemanticError("Expected column reference in NULL check");
    }

    const std::string columnName = expr->expr->name;
    if (!table->hasColumn(columnName))
    {
        throw SemanticError("Column not found: " + columnName);
    }

    ColumnType type = table->getColumnType(columnName);
    const auto &columnData = table->getData().at(columnName);

    // For each type, define what constitutes a NULL value
    switch (type)
    {
    case ColumnType::STRING:
        return columnData[rowIndex].s == nullptr || columnData[rowIndex].s->empty();
    case ColumnType::INTEGER:
        // Maybe use a special value for NULL integers?
        return false; // Need to define what constitutes NULL for integers
    case ColumnType::DOUBLE:
        return std::isnan(columnData[rowIndex].d);
    case ColumnType::DATETIME:
        return columnData[rowIndex].t == nullptr;
    default:
        return false;
    }
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
        case '\\':
        case '+':
        case '*':
        case '?':
        case '(':
        case ')':
        case '[':
        case ']':
        case '{':
        case '}':
        case '|':
        case '^':
        case '$':
            regexPattern += '\\';
            regexPattern += c;
            break;
        default:
            regexPattern += c;
            break;
        }
    }

    std::regex re(regexPattern, std::regex_constants::icase);
    return std::regex_match(value, re);
}