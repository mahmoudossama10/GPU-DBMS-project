#pragma once
#include "../Utilities/UnionUtils.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>

// Define datetime structure

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
          const std::unordered_map<std::string, std::vector<unionV>> &columnData,
          const std::unordered_map<std::string, ColumnType> &columnTypes);
    Table();
    ~Table(); // Need destructor to handle memory management for string and datetime pointers

    // Accessors
    const std::string &getName() const;
    const std::string &getAlias() const;
    std::string &setAlias(std::string alias);

    const std::vector<std::string> &getHeaders() const;
    const std::unordered_map<std::string, ColumnType> &getColumnTypes() const;
    ColumnType getColumnType(const std::string &columnName) const;

    const std::unordered_map<std::string, std::vector<unionV>> &getData() const;
    std::vector<unionV> getRow(int i) const;

    // Convenience methods for getting typed data
    std::string getString(const std::string &columnName, int rowIndex) const;
    int64_t getInteger(const std::string &columnName, int rowIndex) const;
    double getDouble(const std::string &columnName, int rowIndex) const;
    const dateTime &getDateTime(const std::string &columnName, int rowIndex) const;

    const int getSize() const;
    const std::vector<std::string> &getPrimaryKeys() const;
    const std::unordered_map<std::string, ForeignKeyInfo> &getForeignKeys() const;

    // Modified add methods for the new data type
    void addRow(const std::vector<unionV> &row);
    void addColumn(const std::string &columnName, const std::vector<unionV> &columnValues, ColumnType columnType);

    size_t getColumnIndex(const std::string &columnName) const;
    bool hasColumn(const std::string &columnName) const;

    // Get integer column as a convenience (cached)
    const std::vector<int> &getIntColumn(const std::string &colName) const;
    std::string tableName;
    std::unordered_map<std::string, std::vector<unionV>> columnData; // Column-major data
    std::vector<std::string> headers;

private:
    void processHeaders(const std::vector<std::string> &originalHeaders);
    std::string cleanPrimaryKeyHeader(std::string header);
    ForeignKeyInfo parseForeignKeyHeader(const std::string &header);

    // Helper methods for memory management
    void freeUnionMemory(unionV &value, ColumnType type);
    unionV copyUnionValue(const unionV &value, ColumnType type);

    // Convert string to appropriate union value based on type
    unionV stringToUnion(const std::string &str, ColumnType type);

    std::string alias;
    std::unordered_map<std::string, ColumnType> columnTypes;
    std::unordered_map<std::string, size_t> columnIndices;
    std::vector<std::string> primaryKeys;
    std::unordered_map<std::string, ForeignKeyInfo> foreignKeys;
    mutable std::unordered_map<std::string, std::vector<int>> int_column_cache;
};