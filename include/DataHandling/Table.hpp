#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>

struct ForeignKeyInfo
{
    std::string referencedTable;
    std::string referencedColumn;
};

class Table
{
public:
    Table(const std::string &name,
          const std::vector<std::string> &originalHeaders,
          const std::vector<std::vector<std::string>> &dataRows);

    // Accessors
    const std::string &getName() const;
    const std::vector<std::string> &getHeaders() const;
    const std::vector<std::vector<std::string>> &getData() const;
    const std::vector<std::string> &getPrimaryKeys() const;
    const std::unordered_map<std::string, ForeignKeyInfo> &getForeignKeys() const;
    size_t getColumnIndex(const std::string &columnName) const;
    bool hasColumn(const std::string &columnName) const;

private:
    void processHeaders(const std::vector<std::string> &originalHeaders);
    std::string cleanPrimaryKeyHeader(std::string header);
    ForeignKeyInfo parseForeignKeyHeader(const std::string &header);

    std::string tableName;
    std::vector<std::string> headers;
    std::vector<std::vector<std::string>> data;
    std::unordered_map<std::string, size_t> columnIndices;
    std::vector<std::string> primaryKeys;
    std::unordered_map<std::string, ForeignKeyInfo> foreignKeys;
};