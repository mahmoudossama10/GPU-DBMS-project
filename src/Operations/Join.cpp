#include "../../include/Operations/Join.hpp"
#include "Utilities/ErrorHandling.hpp"
#include <algorithm>
#include <cctype>
#include <sstream>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <stack>

std::shared_ptr<Table> JoinPlan::execute()
{
    std::string sqlStatement = "";
    if (whereString != "")
    {
        sqlStatement = "SELECT * FROM dummy WHERE " + whereString;
    }
    else
    {
        sqlStatement = "SELECT * FROM dummy ";
    }
    // Parse the SQL statement
    hsql::SQLParserResult result;
    hsql::SQLParser::parse(sqlStatement, &result);

    const auto *stmt = result.getStatement(0);

    auto selectStmt = static_cast<const hsql::SelectStatement *>(stmt);
    join_condition_ = selectStmt->whereClause;
    // Execute all input plans to get tables
    std::vector<std::shared_ptr<Table>> tables;
    tables.reserve(inputs_.size());

    for (auto &input : inputs_)
    {
        tables.push_back(input->execute());
    }

    // Apply join operation
    return Join::apply(tables, join_condition_);
}

std::shared_ptr<Table> Join::apply(
    const std::vector<std::shared_ptr<Table>> &tables,
    const hsql::Expr *condition)
{
    auto startTime = std::chrono::high_resolution_clock::now();

    if (tables.empty())
    {
        throw SemanticError("No tables provided for join operation");
    }

    if (tables.size() == 1)
    {
        // Just one table, no need to join
        return tables[0];
    }

    // Combine headers from all tables
    std::vector<std::string> headers = combineHeaders(tables);

    // Create a map to store column types for the result table
    std::unordered_map<std::string, ColumnType> columnTypes;

    // Populate column types
    int colOffset = 0;
    for (const auto &table : tables)
    {
        const auto &tableHeaders = table->getHeaders();
        const auto &tableTypes = table->getColumnTypes();

        for (const auto &header : tableHeaders)
        {
            std::string resultHeader = headers[colOffset];
            auto it = tableTypes.find(header);
            if (it != tableTypes.end())
            {
                columnTypes[resultHeader] = it->second;
            }
            else
            {
                // Default to string if type not known
                columnTypes[resultHeader] = ColumnType::STRING;
            }
            colOffset++;
        }
    }

    // Create column data structure for the joined table
    std::unordered_map<std::string, std::vector<unionV>> columnData;
    for (const auto &header : headers)
    {
        columnData[header] = std::vector<unionV>();
    }

    // Calculate the cartesian product size to reserve memory
    size_t estimatedSize = 1;
    for (const auto &table : tables)
    {
        estimatedSize *= table->getSize();
    }

    // Cap the estimate to avoid excessive memory allocation
    const size_t MAX_RESERVE = 1000000;
    estimatedSize = std::min(estimatedSize, MAX_RESERVE);

    for (auto &column : columnData)
    {
        column.second.reserve(estimatedSize);
    }

    // Start with first table's row indices
    std::vector<size_t> currentIndices(tables.size(), 0);

    // Total count of combinations to process
    size_t totalCombinations = 1;
    for (const auto &table : tables)
    {
        totalCombinations *= table->getSize();
    }

    std::cout << "Processing " << totalCombinations << " potential combinations..." << std::endl;

    // Process all combinations
    size_t matchCount = 0;
    size_t processedCount = 0;
    const size_t reportInterval = std::max<size_t>(1, totalCombinations / 10);

    while (true)
    {
        // Evaluate the join condition for the current combination
        if (!condition || evaluateJoinCondition(tables, currentIndices, condition))
        {
            // This combination satisfies the join condition
            matchCount++;

            if (PlanBuilder::joinPlansCount == 1)
            {
                if (PlanBuilder::output_join_table)
                {
                    // Add rows to existing output table
                    int headerIndex = 0;
                    for (size_t t = 0; t < tables.size(); t++)
                    {
                        const auto &table = tables[t];
                        const auto &tableHeaders = table->getHeaders();

                        for (const auto &header : tableHeaders)
                        {
                            // Get the value from the source table
                            ColumnType type = table->getColumnType(header);
                            const auto &sourceData = table->getData().at(header);
                            unionV value = sourceData[currentIndices[t]];

                            // Create a deep copy of the union value if needed
                            switch (type)
                            {
                            case ColumnType::STRING:
                            {
                                std::string *newStr = new std::string(*(value.s));
                                unionV newValue;
                                newValue.s = newStr;
                                PlanBuilder::output_join_table->columnData[headers[headerIndex]].push_back(newValue);
                            }
                            break;
                            case ColumnType::DATETIME:
                            {
                                dateTime *newTime = new dateTime(*(value.t));
                                unionV newValue;
                                newValue.t = newTime;
                                PlanBuilder::output_join_table->columnData[headers[headerIndex]].push_back(newValue);
                            }
                            break;
                            case ColumnType::INTEGER:
                            case ColumnType::DOUBLE:
                                // These don't need deep copy
                                PlanBuilder::output_join_table->columnData[headers[headerIndex]].push_back(value);
                                break;
                            }

                            headerIndex++;
                        }
                    }
                }
                else
                {
                    // Add rows to local columnData map
                    int headerIndex = 0;
                    for (size_t t = 0; t < tables.size(); t++)
                    {
                        const auto &table = tables[t];
                        const auto &tableHeaders = table->getHeaders();

                        for (const auto &header : tableHeaders)
                        {
                            // Get the value from the source table
                            ColumnType type = table->getColumnType(header);
                            const auto &sourceData = table->getData().at(header);
                            unionV value = sourceData[currentIndices[t]];

                            // Create a deep copy of the union value if needed
                            switch (type)
                            {
                            case ColumnType::STRING:
                            {
                                std::string *newStr = new std::string(*(value.s));
                                unionV newValue;
                                newValue.s = newStr;
                                columnData[headers[headerIndex]].push_back(newValue);
                            }
                            break;
                            case ColumnType::DATETIME:
                            {
                                dateTime *newTime = new dateTime(*(value.t));
                                unionV newValue;
                                newValue.t = newTime;
                                columnData[headers[headerIndex]].push_back(newValue);
                            }
                            break;
                            case ColumnType::INTEGER:
                            case ColumnType::DOUBLE:
                                // These don't need deep copy
                                columnData[headers[headerIndex]].push_back(value);
                                break;
                            }

                            headerIndex++;
                        }
                    }

                    // Create result table with appropriate column types
                    std::unordered_map<std::string, ColumnType> columnTypes;

                    int colOffset = 0;
                    for (const auto &table : tables)
                    {
                        const auto &tableHeaders = table->getHeaders();
                        const auto &tableTypes = table->getColumnTypes();

                        for (const auto &header : tableHeaders)
                        {
                            std::string resultHeader = headers[colOffset];
                            auto it = tableTypes.find(header);
                            if (it != tableTypes.end())
                            {
                                columnTypes[resultHeader] = it->second;
                            }
                            else
                            {
                                // Default to string if type not known
                                columnTypes[resultHeader] = ColumnType::STRING;
                            }
                            colOffset++;
                        }
                    }

                    PlanBuilder::output_join_table = std::make_shared<Table>("joined_result", headers, columnData, columnTypes);
                }
            }
            else
            {
                // Add rows to local columnData map
                int headerIndex = 0;
                for (size_t t = 0; t < tables.size(); t++)
                {
                    const auto &table = tables[t];
                    const auto &tableHeaders = table->getHeaders();

                    for (const auto &header : tableHeaders)
                    {
                        // Get the value from the source table
                        ColumnType type = table->getColumnType(header);
                        const auto &sourceData = table->getData().at(header);
                        unionV value = sourceData[currentIndices[t]];

                        // Create a deep copy of the union value if needed
                        switch (type)
                        {
                        case ColumnType::STRING:
                        {
                            std::string *newStr = new std::string(*(value.s));
                            unionV newValue;
                            newValue.s = newStr;
                            columnData[headers[headerIndex]].push_back(newValue);
                        }
                        break;
                        case ColumnType::DATETIME:
                        {
                            dateTime *newTime = new dateTime(*(value.t));
                            unionV newValue;
                            newValue.t = newTime;
                            columnData[headers[headerIndex]].push_back(newValue);
                        }
                        break;
                        case ColumnType::INTEGER:
                        case ColumnType::DOUBLE:
                            // These don't need deep copy
                            columnData[headers[headerIndex]].push_back(value);
                            break;
                        }

                        headerIndex++;
                    }
                }

                // Create result table with appropriate column types
                std::unordered_map<std::string, ColumnType> columnTypes;

                int colOffset = 0;
                for (const auto &table : tables)
                {
                    const auto &tableHeaders = table->getHeaders();
                    const auto &tableTypes = table->getColumnTypes();

                    for (const auto &header : tableHeaders)
                    {
                        std::string resultHeader = headers[colOffset];
                        auto it = tableTypes.find(header);
                        if (it != tableTypes.end())
                        {
                            columnTypes[resultHeader] = it->second;
                        }
                        else
                        {
                            // Default to string if type not known
                            columnTypes[resultHeader] = ColumnType::STRING;
                        }
                        colOffset++;
                    }
                }
            }
        }

        // Update progress periodically
        processedCount++;
        if (processedCount % reportInterval == 0)
        {
            std::cout << "Processed " << processedCount << "/" << totalCombinations
                      << " combinations (" << (processedCount * 100.0 / totalCombinations)
                      << "%), matches found: " << matchCount << std::endl;
        }

        // Move to the next combination
        int tableToIncrement = tables.size() - 1;
        while (tableToIncrement >= 0)
        {
            currentIndices[tableToIncrement]++;
            if (currentIndices[tableToIncrement] < tables[tableToIncrement]->getSize())
            {
                break;
            }

            // Reset this index and increment the next table's index
            currentIndices[tableToIncrement] = 0;
            tableToIncrement--;
        }

        // If we've processed all combinations, we're done
        if (tableToIncrement < 0)
        {
            break;
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> processingTime = endTime - startTime;

    std::cout << "Join completed in " << processingTime.count() << " ms, found "
              << matchCount << " matching rows." << std::endl;

    // Create and return the final joined table

    PlanBuilder::joinPlansCount--;

    if (PlanBuilder::joinPlansCount == 0)
    {
        if (PlanBuilder::output_join_table)
        {
            return PlanBuilder::output_join_table;
        }
        else
        {
            return std::make_shared<Table>("joined_result", headers, columnData, columnTypes);
        }
    }
    else
    {
        return std::make_shared<Table>("joined_result", headers, columnData, columnTypes);
    }
}

bool Join::evaluateJoinCondition(
    const std::vector<std::shared_ptr<Table>> &tables,
    const std::vector<size_t> &rowIndices,
    const hsql::Expr *condition)
{
    if (!condition)
    {
        return true; // No condition means cross join - all combinations match
    }

    switch (condition->type)
    {
    case hsql::kExprOperator:
        return handleLogicalOperator(tables, rowIndices, condition);

    default:
        throw SemanticError("Unsupported condition type in join condition: " +
                            std::to_string(condition->type));
    }
}

bool Join::handleLogicalOperator(
    const std::vector<std::shared_ptr<Table>> &tables,
    const std::vector<size_t> &rowIndices,
    const hsql::Expr *expr)
{
    switch (expr->opType)
    {
    // Logical operators
    case hsql::kOpAnd:
        return evaluateJoinCondition(tables, rowIndices, expr->expr) &&
               evaluateJoinCondition(tables, rowIndices, expr->expr2);

    case hsql::kOpOr:
        return evaluateJoinCondition(tables, rowIndices, expr->expr) ||
               evaluateJoinCondition(tables, rowIndices, expr->expr2);

    case hsql::kOpNot:
        return !evaluateJoinCondition(tables, rowIndices, expr->expr);

    // Comparison operators
    case hsql::kOpEquals:
    case hsql::kOpNotEquals:
    case hsql::kOpLess:
    case hsql::kOpLessEq:
    case hsql::kOpGreater:
    case hsql::kOpGreaterEq:
        return handleComparison(tables, rowIndices, expr);

    default:
        throw SemanticError("Unsupported operator in join condition: " +
                            std::to_string(expr->opType));
    }
}

bool Join::handleComparison(
    const std::vector<std::shared_ptr<Table>> &tables,
    const std::vector<size_t> &rowIndices,
    const hsql::Expr *expr)
{
    // We need to find which tables and columns are involved
    int leftTableIdx, rightTableIdx;
    ColumnType leftType, rightType;
    unionV leftValue, rightValue;

    // Get values from the appropriate tables
    leftValue = getExprValue(tables, rowIndices, expr->expr, leftType, leftTableIdx);
    rightValue = getExprValue(tables, rowIndices, expr->expr2, rightType, rightTableIdx);

    // Compare the values
    return compareValues(leftValue, leftType, rightValue, rightType, expr->opType);
}

unionV Join::getExprValue(
    const std::vector<std::shared_ptr<Table>> &tables,
    const std::vector<size_t> &rowIndices,
    const hsql::Expr *expr,
    ColumnType &outType,
    int &tableIndex)
{
    unionV result;
    tableIndex = -1;

    if (expr->type == hsql::kExprColumnRef)
    {
        // Find which table this column belongs to
        auto [foundTableIdx, columnName] = findTableAndColumn(tables, expr->name, expr->table);

        if (foundTableIdx == -1 || columnName.empty())
        {
            throw SemanticError("Column not found: " +
                                std::string(expr->table ? expr->table : "") +
                                (expr->table ? "." : "") + expr->name);
        }

        tableIndex = foundTableIdx;
        outType = tables[foundTableIdx]->getColumnType(columnName);
        const auto &columnData = tables[foundTableIdx]->getData().at(columnName);
        return columnData[rowIndices[foundTableIdx]];
    }
    else if (expr->type == hsql::kExprLiteralInt)
    {
        outType = ColumnType::INTEGER;
        result.i->value = expr->ival;
    }
    else if (expr->type == hsql::kExprLiteralFloat)
    {
        outType = ColumnType::DOUBLE;
        result.d->value = expr->fval;
    }
    else if (expr->type == hsql::kExprLiteralString)
    {
        outType = ColumnType::STRING;
        result.s = new std::string(expr->name);
    }
    else
    {
        throw SemanticError("Unsupported expression type in join condition: " +
                            std::to_string(expr->type));
    }

    return result;
}
std::pair<int, std::string> Join::findTableAndColumn(
    const std::vector<std::shared_ptr<Table>> &tables,
    const char *columnName,
    const char *tableName)
{
    if (!columnName)
    {
        return {-1, ""};
    }

    std::string colName = columnName;

    for (size_t i = 0; i < tables.size(); i++)
    {
        const auto &table = tables[i];
        const auto &headers = table->getHeaders();

        for (const auto &header : headers)
        {
            if (tableName && tableName[0] != '\0')
            {
                std::string fullColumn = std::string(tableName) + "." + colName;

                if (header == fullColumn ||
                    (header == colName && table->getAlias() == tableName) ||
                    (header == colName && table->getName() == tableName))
                {
                    return {static_cast<int>(i), header};
                }
            }
            else
            {
                if (header == colName)
                {
                    return {static_cast<int>(i), header};
                }
            }
        }
    }

    return {-1, ""};
}

bool Join::compareValues(
    const unionV &lhs, ColumnType lhsType,
    const unionV &rhs, ColumnType rhsType,
    hsql::OperatorType op)
{
    // If types don't match, convert to the more precise type
    if (lhsType != rhsType)
    {
        // Handle type conversion for comparison
        if ((lhsType == ColumnType::INTEGER && rhsType == ColumnType::DOUBLE) ||
            (lhsType == ColumnType::DOUBLE && rhsType == ColumnType::INTEGER))
        {
            // Convert to double comparison
            double lhsDouble = (lhsType == ColumnType::INTEGER) ? static_cast<double>(lhs.i->value) : lhs.d->value;
            double rhsDouble = (rhsType == ColumnType::INTEGER) ? static_cast<double>(rhs.i->value) : rhs.d->value;
            return compareDoubles(lhsDouble, rhsDouble, op);
        }
        else
        {
            // For other mixed types, convert both to strings and compare as strings
            std::string lhsStr, rhsStr;

            switch (lhsType)
            {
            case ColumnType::STRING:
                lhsStr = *(lhs.s);
                break;
            case ColumnType::INTEGER:
            {
                std::stringstream ss;
                ss << lhs.i->value;
                lhsStr = ss.str();
                break;
            }
            case ColumnType::DOUBLE:
            {
                std::stringstream ss;
                ss << lhs.d->value;
                lhsStr = ss.str();
                break;
            }
            case ColumnType::DATETIME:
            {
                std::stringstream ss;
                const dateTime &dt = *(lhs.t);
                ss << dt.year << "-";
                ss << std::setw(2) << std::setfill('0') << dt.month << "-";
                ss << std::setw(2) << std::setfill('0') << dt.day << " ";
                ss << std::setw(2) << std::setfill('0') << (int)dt.hour << ":";
                ss << std::setw(2) << std::setfill('0') << (int)dt.minute << ":";
                ss << std::setw(2) << std::setfill('0') << (int)dt.second;
                lhsStr = ss.str();
                break;
            }
            }

            switch (rhsType)
            {
            case ColumnType::STRING:
                rhsStr = *(rhs.s);
                break;
            case ColumnType::INTEGER:
            {
                std::stringstream ss;
                ss << rhs.i->value;
                rhsStr = ss.str();
                break;
            }
            case ColumnType::DOUBLE:
            {
                std::stringstream ss;
                ss << rhs.d->value;
                rhsStr = ss.str();
                break;
            }
            case ColumnType::DATETIME:
            {
                std::stringstream ss;
                const dateTime &dt = *(rhs.t);
                ss << dt.year << "-";
                ss << std::setw(2) << std::setfill('0') << dt.month << "-";
                ss << std::setw(2) << std::setfill('0') << dt.day << " ";
                ss << std::setw(2) << std::setfill('0') << (int)dt.hour << ":";
                ss << std::setw(2) << std::setfill('0') << (int)dt.minute << ":";
                ss << std::setw(2) << std::setfill('0') << (int)dt.second;
                rhsStr = ss.str();
                break;
            }
            }

            return compareStrings(lhsStr, rhsStr, op);
        }
    }

    // Same types, perform type-specific comparison
    switch (lhsType)
    {
    case ColumnType::STRING:
        return compareStrings(*(lhs.s), *(rhs.s), op);

    case ColumnType::INTEGER:
        return compareIntegers(lhs.i->value, rhs.i->value, op);

    case ColumnType::DOUBLE:
        return compareDoubles(lhs.d->value, rhs.d->value, op);

    default:
        throw SemanticError("Unsupported data type in join comparison");
    }
}

bool Join::compareStrings(const std::string &lhs, const std::string &rhs, hsql::OperatorType op)
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
        throw SemanticError("Unsupported string comparison operator in join");
    }
}

bool Join::compareIntegers(int64_t lhs, int64_t rhs, hsql::OperatorType op)
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
        throw SemanticError("Unsupported integer comparison operator in join");
    }
}

bool Join::compareDoubles(double lhs, double rhs, hsql::OperatorType op)
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
        throw SemanticError("Unsupported double comparison operator in join");
    }
}

std::vector<std::string> Join::combineHeaders(const std::vector<std::shared_ptr<Table>> &tables)
{
    std::vector<std::string> headers;

    for (const auto &table : tables)
    {
        const auto &tableHeaders = table->getHeaders();
        const std::string &alias = table->getAlias();

        for (const auto &header : tableHeaders)
        {
            // Use alias.column notation if alias exists
            if (!alias.empty())
            {
                headers.push_back(alias + "." + header);
            }
            else
            {
                headers.push_back(header);
            }
        }
    }

    return headers;
}