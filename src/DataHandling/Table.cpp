#include "../../include/DataHandling/Table.hpp"
#include "Utilities/StringUtils.hpp"
#include <algorithm>

Table::Table(const std::string &name,
             const std::vector<std::string> &originalHeaders,
             const std::vector<std::vector<std::string>> &dataRows)
    : tableName(name), data(dataRows)
{
    processHeaders(originalHeaders);
}

void Table::processHeaders(const std::vector<std::string> &originalHeaders)
{
    for (const auto &header : originalHeaders)
    {
        std::string cleanedName;
        bool isPrimary = false;
        bool isForeign = false;
        ForeignKeyInfo fkInfo;

        // Check for primary key
        if (header.find("(P)") != std::string::npos)
        {
            cleanedName = cleanPrimaryKeyHeader(header);
            primaryKeys.push_back(cleanedName);
            isPrimary = true;
        }
        // Check for foreign key
        else if (StringUtils::startsWith(header, "#"))
        {
            fkInfo = parseForeignKeyHeader(header);
            cleanedName = fkInfo.referencedColumn;
            foreignKeys[cleanedName] = fkInfo;
            isForeign = true;
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

void Table::addRow(std::vector<std::string> &row)
{
    if (row.size() != headers.size())
    {
        throw std::invalid_argument(
            "Row size (" + std::to_string(row.size()) +
            ") doesn't match table columns (" +
            std::to_string(headers.size()) + ")");
    }
    data.push_back(row);
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
const std::vector<std::string> &Table::getRow(int i) const { return data[i]; }

std::string &Table::setAlias(std::string alias)
{
    this->alias = alias;
}

const std::vector<std::string> &Table::getHeaders() const { return headers; }
const std::vector<std::vector<std::string>> &Table::getData() const { return data; }
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
    return data.size();
}

const std::vector<int>& Table::getIntColumn(const std::string& colName) const {
    auto it = int_column_cache.find(colName);
    if (it != int_column_cache.end()) {
        return it->second;
    }
    // Parse from string data
    size_t idx = getColumnIndex(colName);
    std::vector<int> values;
    values.reserve(getSize());
    for (const auto& row : data) {
        try {
            values.push_back(std::stoi(row[idx]));
        } catch (...) {
            throw std::runtime_error("Non-integer value found in column '" + colName + "'");
        }
    }
    // Save and return reference
    auto insert_result = int_column_cache.emplace(colName, std::move(values));
    return insert_result.first->second;
}