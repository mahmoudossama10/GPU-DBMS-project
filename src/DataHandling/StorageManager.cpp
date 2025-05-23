#include "DataHandling/StorageManager.hpp"
#include "DataHandling/CSVProcessor.hpp"
#include <stdexcept>

void StorageManager::loadTable(const std::string &tableName, const std::string &filepath)
{
    if (tables.find(tableName) != tables.end())
    {
        throw std::runtime_error("Table already loaded: " + tableName);
    }

    auto [headers, data, columnTypes] = CSVProcessor::loadCSV(filepath);
    tables[tableName] = std::make_unique<Table>(tableName, headers, data, columnTypes);
}

Table &StorageManager::getTable(const std::string &tableName)
{
    auto it = tables.find(tableName);
    if (it == tables.end())
    {
        throw std::runtime_error("Table not found: " + tableName);
    }
    return *(it->second);
}

bool StorageManager::tableExists(const std::string &tableName) const
{
    return tables.find(tableName) != tables.end();
}

std::vector<std::string> StorageManager::listTables() const
{
    std::vector<std::string> tableNames;
    for (const auto &pair : tables)
    {
        tableNames.push_back(pair.first);
    }
    return tableNames;
}