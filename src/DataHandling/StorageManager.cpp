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

// Add this implementation to the StorageManager.cpp file:
void StorageManager::addTable(const std::shared_ptr<Table> &table)
{
    if (!table)
    {
        throw std::invalid_argument("Cannot add null table");
    }

    const std::string &tableName = table->tableName;

    // Check if a table with this name already exists
    if (tables.find(tableName) != tables.end())
    {
        throw std::runtime_error("Table already exists: " + tableName);
    }

    // Create a new unique_ptr by cloning the table data
    // This is necessary because we're converting from shared_ptr to unique_ptr
    tables[tableName] = std::make_unique<Table>(
        tableName,
        table->headers,
        table->getData(),
        table->getColumnTypes());
}

void StorageManager::renameTable(const std::string &oldName, const std::string &newName)
{
    // Check if old table exists
    auto it = tables.find(oldName);
    if (it == tables.end())
    {
        throw std::runtime_error("Table not found: " + oldName);
    }

    // Check if new table name already exists
    if (tables.find(newName) != tables.end())
    {
        throw std::runtime_error("Table already exists: " + newName);
    }

    // Save the pointer to the table
    std::unique_ptr<Table> table = std::move(it->second);

    // Update the table's internal name
    table->tableName = newName;

    // Remove old entry and add new one
    tables.erase(it);
    tables[newName] = std::move(table);
}

std::vector<std::string> StorageManager::getTableNames() const
{
    std::vector<std::string> names;
    names.reserve(tables.size());

    for (const auto &[name, _] : tables)
    {
        names.push_back(name);
    }

    return names;
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