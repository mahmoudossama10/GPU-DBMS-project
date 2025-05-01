#include "../../include/Operations/Join.hpp"
#include "../../include/Utilities/ErrorHandling.hpp"
#include <algorithm>

JoinPlan::JoinPlan(std::unique_ptr<ExecutionPlan> left,
                   std::unique_ptr<ExecutionPlan> right,
                   const std::string &left_alias,
                   const std::string &right_alias)
    : left_(std::move(left)),
      right_(std::move(right)),
      left_alias_(left_alias),
      right_alias_(right_alias) {}

std::shared_ptr<Table> JoinPlan::execute()
{
    auto left_table = left_->execute();
    auto right_table = right_->execute();

    // Combine headers
    auto headers = combineHeaders(*left_table, *right_table);

    // Create column types map for the result table
    std::unordered_map<std::string, ColumnType> resultColumnTypes;

    // Add left table's column types with prefixed headers
    const auto &leftColumnTypes = left_table->getColumnTypes();
    for (const auto &[colName, colType] : leftColumnTypes)
    {
        resultColumnTypes[prefixHeader(colName, left_alias_)] = colType;
    }

    // Add right table's column types with prefixed headers
    const auto &rightColumnTypes = right_table->getColumnTypes();
    for (const auto &[colName, colType] : rightColumnTypes)
    {
        resultColumnTypes[prefixHeader(colName, right_alias_)] = colType;
    }

    // Compute product with the new unionV column data structure
    auto columnData = computeProduct(*left_table, *right_table, resultColumnTypes);

    return std::make_shared<Table>(
        "JoinedTable",
        headers,
        columnData,
        resultColumnTypes);
}

std::vector<std::string> JoinPlan::combineHeaders(
    const Table &left,
    const Table &right) const
{
    std::vector<std::string> headers;

    // Process left headers
    for (const auto &h : left.getHeaders())
    {
        headers.push_back(prefixHeader(h, left_alias_));
    }

    // Process right headers
    for (const auto &h : right.getHeaders())
    {
        headers.push_back(prefixHeader(h, right_alias_));
    }

    return headers;
}

std::unordered_map<std::string, std::vector<unionV>> JoinPlan::computeProduct(
    const Table &left,
    const Table &right,
    std::unordered_map<std::string, ColumnType> &resultColumnTypes) const
{
    std::unordered_map<std::string, std::vector<unionV>> result;

    // Get column-major data from both tables
    const auto &leftData = left.getData();
    const auto &leftColumnTypes = left.getColumnTypes();
    const auto &rightData = right.getData();
    const auto &rightColumnTypes = right.getColumnTypes();

    // Determine row count by checking size of tables
    size_t leftRowCount = left.getSize();
    size_t rightRowCount = right.getSize();

    // If either table is empty, return empty result
    if (leftRowCount == 0 || rightRowCount == 0)
    {
        return result;
    }

    // Calculate total rows in result (product of the two tables)
    size_t totalRows = leftRowCount * rightRowCount;

    // Initialize result columns with the right size
    for (const auto &h : left.getHeaders())
    {
        std::string prefixedHeader = prefixHeader(h, left_alias_);
        result[prefixedHeader] = std::vector<unionV>(totalRows);
    }

    for (const auto &h : right.getHeaders())
    {
        std::string prefixedHeader = prefixHeader(h, right_alias_);
        result[prefixedHeader] = std::vector<unionV>(totalRows);
    }

    // Index for tracking position in result
    size_t resultIdx = 0;

    // Cartesian product calculation
    for (size_t l_row = 0; l_row < leftRowCount; ++l_row)
    {
        for (size_t r_row = 0; r_row < rightRowCount; ++r_row)
        {
            // Add all left columns for this row
            for (const auto &colName : left.getHeaders())
            {
                std::string prefixedHeader = prefixHeader(colName, left_alias_);
                const auto &colData = leftData.at(colName);
                ColumnType colType = leftColumnTypes.at(colName);

                // Deep copy the union value based on its type
                switch (colType)
                {
                case ColumnType::STRING:
                    result[prefixedHeader][resultIdx].s = new std::string(*colData[l_row].s);
                    break;
                case ColumnType::INTEGER:
                    result[prefixedHeader][resultIdx].i = colData[l_row].i;
                    break;
                case ColumnType::DOUBLE:
                    result[prefixedHeader][resultIdx].d = colData[l_row].d;
                    break;
                case ColumnType::DATETIME:
                {
                    dateTime *newDateTime = new dateTime;
                    *newDateTime = *colData[l_row].t;
                    result[prefixedHeader][resultIdx].t = newDateTime;
                    break;
                }
                }
            }

            // Add all right columns for this row
            for (const auto &colName : right.getHeaders())
            {
                std::string prefixedHeader = prefixHeader(colName, right_alias_);
                const auto &colData = rightData.at(colName);
                ColumnType colType = rightColumnTypes.at(colName);

                // Deep copy the union value based on its type
                switch (colType)
                {
                case ColumnType::STRING:
                    result[prefixedHeader][resultIdx].s = new std::string(*colData[r_row].s);
                    break;
                case ColumnType::INTEGER:
                    result[prefixedHeader][resultIdx].i = colData[r_row].i;
                    break;
                case ColumnType::DOUBLE:
                    result[prefixedHeader][resultIdx].d = colData[r_row].d;
                    break;
                case ColumnType::DATETIME:
                {
                    dateTime *newDateTime = new dateTime;
                    *newDateTime = *colData[r_row].t;
                    result[prefixedHeader][resultIdx].t = newDateTime;
                    break;
                }
                }
            }

            resultIdx++;
        }
    }

    return result;
}

std::string JoinPlan::prefixHeader(const std::string &header,
                                   const std::string &alias) const
{
    if (!alias.empty())
    {
        return alias + "." + header;
    }
    return header;
}