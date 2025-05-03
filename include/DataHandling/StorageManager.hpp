#pragma once
#include <unordered_map>
#include <memory>
#include "Table.hpp"

class StorageManager
{
public:
    void loadTable(const std::string &tableName, const std::string &filepath);
    Table &getTable(const std::string &tableName);
    bool tableExists(const std::string &tableName) const;
    std::vector<std::string> listTables() const;

private:
    std::unordered_map<std::string, std::unique_ptr<Table>> tables;
};