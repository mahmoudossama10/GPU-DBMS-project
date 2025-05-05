#include "../../include/DataHandling/Table.hpp"
#include "Utilities/StringUtils.hpp"
#include <algorithm>
#include <cstring>

// Constructor that takes union data
Table::Table(const std::string &name,
             const std::vector<std::string> &originalHeaders,
             const std::unordered_map<std::string, std::vector<unionV>> &columnData,
             const std::unordered_map<std::string, ColumnType> &columnTypes)
    : tableName(name), columnTypes(columnTypes), columnData(columnData)
{
    processHeaders(originalHeaders);

    // Validate that all columns have the same size and that we have types for each column
    if (!headers.empty())
    {
        if (headers.size() != columnTypes.size())
        {
            throw std::invalid_argument("Column headers and column types count mismatch");
        }

        size_t expectedSize = columnData.at(headers[0]).size();
        for (const auto &header : headers)
        {
            auto it = columnData.find(header);
            if (it == columnData.end())
            {
                throw std::invalid_argument("Column missing from data: " + header);
            }
            if (it->second.size() != expectedSize)
            {
                throw std::invalid_argument("Column size mismatch for: " + header);
            }
        }
    }
}

Table::Table()
{
    tableName = "";
}
// Destructor to clean up memory
Table::~Table()
{
    // Free memory for string and datetime pointers
    // for (size_t colIdx = 0; colIdx < headers.size(); ++colIdx)
    // {
    //     const std::string &header = headers[colIdx];
    //     ColumnType type = columnTypes.at(headers[colIdx]);

    //     auto &values = columnData[header];
    //     for (auto &value : values)
    //     {
    //         freeUnionMemory(value, type);
    //     }
    // }
}

void Table::freeUnionMemory(unionV &value, ColumnType type)
{
    if (type == ColumnType::STRING && value.s != nullptr)
    {
        delete value.s;
        value.s = nullptr;
    }
    else if (type == ColumnType::DATETIME && value.t != nullptr)
    {
        delete value.t;
        value.t = nullptr;
    }
}

unionV Table::copyUnionValue(const unionV &value, ColumnType type)
{
    unionV copy;

    switch (type)
    {
    case ColumnType::STRING:
        copy.s = (value.s != nullptr) ? new std::string(*value.s) : nullptr;
        break;
    case ColumnType::INTEGER:
        copy.i = value.i;
        break;
    case ColumnType::DOUBLE:
        copy.d = value.d;
        break;
    case ColumnType::DATETIME:
        if (value.t != nullptr)
        {
            copy.t = new dateTime;
            *copy.t = *value.t;
        }
        else
        {
            copy.t = nullptr;
        }
        break;
    }

    return copy;
}

unionV Table::stringToUnion(const std::string &str, ColumnType type)
{
    unionV result;

    switch (type)
    {
    case ColumnType::STRING:
        result.s = new std::string(str);
        break;
    case ColumnType::INTEGER:
        try
        {
            result.i = std::stoll(str);
        }
        catch (...)
        {
            throw std::runtime_error("Invalid integer format: " + str);
        }
        break;
    case ColumnType::DOUBLE:
        try
        {
            result.d = std::stod(str);
        }
        catch (...)
        {
            throw std::runtime_error("Invalid double format: " + str);
        }
        break;
    case ColumnType::DATETIME:
        // Parse YYYY-MM-DD HH:MM:SS format or similar
        try
        {
            result.t = new dateTime;

            // Very basic parsing - production code would need more robust parsing
            std::string datePart = str.substr(0, 10); // YYYY-MM-DD
            std::string timePart = (str.length() > 11) ? str.substr(11) : "00:00:00";

            result.t->year = static_cast<unsigned short>(std::stoi(datePart.substr(0, 4)));
            result.t->month = static_cast<unsigned short>(std::stoi(datePart.substr(5, 2)));
            result.t->day = static_cast<unsigned short>(std::stoi(datePart.substr(8, 2)));

            result.t->hour = static_cast<unsigned char>(std::stoi(timePart.substr(0, 2)));
            result.t->minute = static_cast<unsigned char>(std::stoi(timePart.substr(3, 2)));
            result.t->second = static_cast<unsigned char>(std::stoi(timePart.substr(6, 2)));
        }
        catch (...)
        {
            throw std::runtime_error("Invalid datetime format: " + str +
                                     " (expected YYYY-MM-DD HH:MM:SS)");
        }
        break;
    }

    return result;
}

void Table::addColumn(const std::string &columnName, const std::vector<unionV> &columnValues, ColumnType columnType)
{
    // Check if column already exists
    if (columnData.find(columnName) != columnData.end())
    {
        throw std::runtime_error("Column already exists: " + columnName);
    }

    // Check size consistency if table already has data
    if (!columnData.empty())
    {
        size_t expected_size = columnData.begin()->second.size();
        if (columnValues.size() != expected_size)
        {
            throw std::runtime_error("Column size mismatch. Expected " +
                                     std::to_string(expected_size) + " rows, got " +
                                     std::to_string(columnValues.size()));
        }
    }

    // Add the column data - copy values to avoid memory issues
    std::vector<unionV> valueCopy;
    valueCopy.reserve(columnValues.size());

    for (const auto &value : columnValues)
    {
        valueCopy.push_back(copyUnionValue(value, columnType));
    }

    columnData.at(columnName) = std::move(valueCopy);

    // Update headers and indices if this is a new column
    if (columnIndices.find(columnName) == columnIndices.end())
    {
        headers.push_back(columnName);
        columnTypes.at(columnName) = columnType;
        columnIndices.at(columnName) = headers.size() - 1;
    }

    // Clear any cached int representation
    int_column_cache.erase(columnName);
}

void Table::processHeaders(const std::vector<std::string> &originalHeaders)
{
    for (const auto &header : originalHeaders)
    {
        std::string cleanedName;

        ForeignKeyInfo fkInfo;

        // Check for primary key
        if (header.find("(P)") != std::string::npos)
        {
            cleanedName = cleanPrimaryKeyHeader(header);
            primaryKeys.push_back(cleanedName);
        }
        else
        {
            cleanedName = StringUtils::trim(header);
        }

        // Validate unique column names
        if (columnIndices.find(cleanedName) != columnIndices.end())
        {
            throw std::invalid_argument("Duplicate column name: " + cleanedName);
        }

        // Store cleaned header and index
        headers.push_back(cleanedName);
        columnIndices[cleanedName] = headers.size() - 1;
    }
}

void Table::addRow(const std::vector<unionV> &row)
{

    // Add each value to its respective column vector
    for (size_t i = 0; i < headers.size(); ++i)
    {
        columnData[headers[i]].push_back(copyUnionValue(row[i], columnTypes[headers[i]]));
    }
}

std::string Table::cleanPrimaryKeyHeader(std::string header)
{
    size_t pos = header.find("(P)");
    if (pos != std::string::npos)
    {
        header.erase(pos, 3); // Remove "(P)"
    }
    return StringUtils::trim(header);
}

ForeignKeyInfo Table::parseForeignKeyHeader(const std::string &header)
{
    std::string fkPart = header.substr(1); // Remove leading '#'
    size_t underscorePos = fkPart.find('_');

    if (underscorePos == std::string::npos || underscorePos == 0)
    {
        throw std::invalid_argument("Malformed foreign key header: " + header);
    }

    ForeignKeyInfo info;
    info.referencedTable = fkPart.substr(0, underscorePos);
    info.referencedColumn = fkPart.substr(underscorePos + 1);

    if (info.referencedTable.empty() || info.referencedColumn.empty())
    {
        throw std::invalid_argument("Invalid foreign key format: " + header);
    }

    return info;
}

// Accessor implementations
const std::string &Table::getName() const { return tableName; }

const std::string &Table::getAlias() const { return alias; }

std::string &Table::setAlias(std::string alias)
{
    this->alias = alias;
    return this->alias;
}

const std::vector<std::string> &Table::getHeaders() const { return headers; }

const std::unordered_map<std::string, ColumnType> &Table::getColumnTypes() const { return columnTypes; }

ColumnType Table::getColumnType(const std::string &columnName) const
{
    return columnTypes.at(columnName);
}

const std::unordered_map<std::string, std::vector<unionV>> &Table::getData() const
{
    return columnData;
}

std::vector<unionV> Table::getRow(int i) const
{
    if (i < 0 || i >= getSize())
    {
        throw std::out_of_range("Row index out of range: " + std::to_string(i));
    }

    std::vector<unionV> row;
    row.reserve(headers.size());

    for (const auto &header : headers)
    {
        row.push_back(columnData.at(header)[i]);
    }

    return row;
}

std::string Table::getString(const std::string &columnName, int rowIndex) const
{
    if (!hasColumn(columnName))
    {
        throw std::out_of_range("Column not found: " + columnName);
    }

    ColumnType type = columnTypes.at(columnName);

    if (type != ColumnType::STRING)
    {
        throw std::runtime_error("Column '" + columnName + "' is not a string column");
    }

    const auto &value = columnData.at(columnName)[rowIndex];
    return (value.s != nullptr) ? *value.s : "";
}

int64_t Table::getInteger(const std::string &columnName, int rowIndex) const
{
    if (!hasColumn(columnName))
    {
        throw std::out_of_range("Column not found: " + columnName);
    }

    ColumnType type = columnTypes.at(columnName);

    if (type != ColumnType::INTEGER)
    {
        throw std::runtime_error("Column '" + columnName + "' is not an integer column");
    }

    return columnData.at(columnName)[rowIndex].i;
}

double Table::getDouble(const std::string &columnName, int rowIndex) const
{
    if (!hasColumn(columnName))
    {
        throw std::out_of_range("Column not found: " + columnName);
    }

    ColumnType type = columnTypes.at(columnName);

    if (type != ColumnType::DOUBLE)
    {
        throw std::runtime_error("Column '" + columnName + "' is not a double column");
    }

    return columnData.at(columnName)[rowIndex].d;
}

const dateTime &Table::getDateTime(const std::string &columnName, int rowIndex) const
{
    if (!hasColumn(columnName))
    {
        throw std::out_of_range("Column not found: " + columnName);
    }

    ColumnType type = columnTypes.at(columnName);

    if (type != ColumnType::DATETIME)
    {
        throw std::runtime_error("Column '" + columnName + "' is not a datetime column");
    }

    const auto &value = columnData.at(columnName)[rowIndex];
    if (value.t == nullptr)
    {
        throw std::runtime_error("Null datetime value in column '" + columnName + "' at row " + std::to_string(rowIndex));
    }

    return *value.t;
}

const std::vector<std::string> &Table::getPrimaryKeys() const { return primaryKeys; }

const std::unordered_map<std::string, ForeignKeyInfo> &Table::getForeignKeys() const { return foreignKeys; }

size_t Table::getColumnIndex(const std::string &columnName) const
{
    auto it = columnIndices.find(columnName);
    if (it == columnIndices.end())
    {
        throw std::out_of_range("Column not found: " + columnName);
    }
    return it->second;
}

bool Table::hasColumn(const std::string &columnName) const
{
    return columnIndices.find(columnName) != columnIndices.end();
}

const int Table::getSize() const
{
    if (headers.empty())
    {
        return 0;
    }

    // All columns should have the same size, so we can just check the first one
    return columnData.at(headers[0]).size();
}

const std::vector<int> &Table::getIntColumn(const std::string &colName) const
{
    auto it = int_column_cache.find(colName);
    if (it != int_column_cache.end())
    {
        return it->second;
    }

    // Check if column exists
    if (!hasColumn(colName))
    {
        throw std::out_of_range("Column not found: " + colName);
    }

    ColumnType type = columnTypes.at(colName);

    // Parse from data based on column type
    const auto &columnVector = columnData.at(colName);
    std::vector<int> values;
    values.reserve(columnVector.size());

    for (const auto &unionValue : columnVector)
    {
        try
        {
            switch (type)
            {
            case ColumnType::STRING:
                if (unionValue.s != nullptr)
                {
                    values.push_back(std::stoi(*unionValue.s));
                }
                else
                {
                    values.push_back(0);
                }
                break;
            case ColumnType::INTEGER:
                values.push_back(static_cast<int>(unionValue.i));
                break;
            case ColumnType::DOUBLE:
                values.push_back(static_cast<int>(unionValue.d));
                break;
            case ColumnType::DATETIME:
                // For datetime, we'll use the year as an integer (just as an example)
                if (unionValue.t != nullptr)
                {
                    values.push_back(unionValue.t->year);
                }
                else
                {
                    values.push_back(0);
                }
                break;
            }
        }
        catch (...)
        {
            throw std::runtime_error("Non-integer value found in column '" + colName + "'");
        }
    }

    // Save and return reference
    auto insert_result = int_column_cache.emplace(colName, std::move(values));
    return insert_result.first->second;
}