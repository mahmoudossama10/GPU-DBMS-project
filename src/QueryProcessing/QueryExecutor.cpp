#include "../../include/QueryProcessing/QueryExecutor.hpp"
#include "../../include/QueryProcessing/PlanBuilder.hpp"
#include "Utilities/ErrorHandling.hpp"
#include <iostream>
#include <hsql/util/sqlhelper.h>
#include <regex>

QueryExecutor::QueryExecutor(std::shared_ptr<StorageManager> storage)
    : storage_(storage),
      plan_builder_(std::make_unique<PlanBuilder>(storage)) {}

std::shared_ptr<Table> QueryExecutor::execute(const std::string &query)
{
    hsql::SQLParserResult result;
    hsql::SQLParser::parse(query, &result);
    // hsql::printStatementInfo(result.getStatement(0));
    if (!result.isValid())
    {
        throw SyntaxError("Parse error: " + std::string(result.errorMsg()));
    }

    if (result.size() != 1)
    {
        throw SyntaxError("Only single-statement queries supported");
    }

    const auto *stmt = result.getStatement(0);
    if (stmt->type() != hsql::kStmtSelect)
    {
        throw SyntaxError("Only SELECT statements supported");
    }

    return execute(static_cast<const hsql::SelectStatement *>(stmt), query);
}

std::shared_ptr<Table> QueryExecutor::execute(const hsql::SelectStatement *stmt, const std::string &query)
{
    validateSelectStatement(stmt);

    std::vector<std::string> allTableNamesOriginal;

    std::vector<std::string> allTableNames;

    if (stmt->fromTable->type == hsql::kTableName)
    {
        allTableNamesOriginal.push_back(std::string(stmt->fromTable->name));

        if (stmt->fromTable->alias != nullptr)
        {
            allTableNames.push_back(
                std::string(stmt->fromTable->name) + "_batch_" + stmt->fromTable->alias->name);
            // std::string newName = std::string(stmt->fromTable->name) + "_" + stmt->fromTable->alias->name;
            // char *modifiedName = new char[newName.size() + 1];
            // std::strcpy(modifiedName, newName.c_str());
            // const_cast<hsql::TableRef *>(stmt->fromTable)->name = modifiedName;
        }
        else
            allTableNames.push_back(std::string(stmt->fromTable->name));
    }
    else if (stmt->fromTable->type == hsql::kTableCrossProduct)
    {
        for (auto tbl : *stmt->fromTable->list)
        {
            if (tbl->type == hsql::kTableName)
            {
                allTableNamesOriginal.push_back(std::string(tbl->name));

                if (tbl->alias != nullptr)
                {
                    allTableNames.push_back(std::string(tbl->name) + "_batch_" + tbl->alias->name);
                    // std::string newName = std::string(tbl->name) + "_" + tbl->alias->name;
                    // char *modifiedName = new char[newName.size() + 1];
                    // std::strcpy(modifiedName, newName.c_str());
                    // const_cast<hsql::TableRef *>(tbl)->name = modifiedName;
                }
                else
                    allTableNames.push_back(std::string(tbl->name));
            }
        }
    }

    // Get all table names
    // std::vector<std::string> allTableNames = storage_->getTableNames();

    // Constant for batch size
    const size_t BATCH_SIZE = 20000;

    // Create batched tables from all tables
    for (int i = 0; i < allTableNames.size(); i++)
    {
        // Skip if this is already a batched table (to prevent infinite recursion)
        if (allTableNames[i].find("_batch0_") != std::string::npos)
        {
            continue;
        }

        // Get the original table
        Table &originalTable = storage_->getTable(allTableNamesOriginal[i]);
        const std::vector<std::string> &headers = originalTable.getHeaders();
        const std::unordered_map<std::string, ColumnType> &columnTypes = originalTable.getColumnTypes();
        const std::unordered_map<std::string, std::vector<unionV>> &columnData = originalTable.getData();

        // Calculate number of batches
        size_t totalRows = originalTable.getSize();
        size_t numBatches = (totalRows + BATCH_SIZE - 1) / BATCH_SIZE; // Ceiling division

        // Create batched tables
        for (size_t batchIdx = 0; batchIdx < numBatches; batchIdx++)
        {
            std::string batchTableName = allTableNames[i] + "_batch0_" + std::to_string(batchIdx);
            std::cout << batchTableName << '\n';
            // Skip if this batch already exists
            if (storage_->tableExists(batchTableName))
            {
                continue;
            }

            // Calculate the range for this batch
            size_t startRow = batchIdx * BATCH_SIZE;
            size_t endRow = std::min(startRow + BATCH_SIZE, totalRows);
            size_t batchSize = endRow - startRow;

            // Create column data for this batch
            std::unordered_map<std::string, std::vector<unionV>> batchData;

            for (const auto &header : headers)
            {
                const std::vector<unionV> &originalCol = columnData.at(header);
                std::vector<unionV> batchCol;
                batchCol.reserve(batchSize);

                // Copy just the rows for this batch
                for (size_t row = startRow; row < endRow; row++)
                {
                    batchCol.push_back(copyUnionValue(originalCol[row], columnTypes.at(header)));
                }

                batchData[header] = std::move(batchCol);
            }

            // Create and store the new batch table
            storage_->tables[batchTableName] = std::make_unique<Table>(
                batchTableName, headers, batchData, columnTypes);
        }
    }

    // Process query on batched tables recursively
    std::shared_ptr<Table> result = processBatchedQuery(stmt, query);

    // Cleanup batch tables and restore original tables

    return result;
}

// Helper function to copy union values based on type (similar to the one in your Table class)
unionV QueryExecutor::copyUnionValue(const unionV &value, ColumnType type)
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
std::shared_ptr<Table> QueryExecutor::processBatchedQuery(const hsql::SelectStatement *stmt, const std::string &query)
{
    std::shared_ptr<Table> finalResult = nullptr;

    // Get all batched table names
    std::vector<std::string> batchedTables;
    for (const auto &tableName : storage_->getTableNames())
    {
        if (tableName.find("_batch0_") != std::string::npos)
        {
            batchedTables.push_back(tableName);
        }
    }

    // If no batched tables, execute the query on the original data
    if (batchedTables.empty())
    {
        return plan_builder_->build(stmt, query)->execute();
    }

    // Group batches by their original table name
    std::map<std::string, std::vector<std::string>> tableGroups;
    for (const auto &batchName : batchedTables)
    {

        std::string originalTableName = batchName.substr(0, batchName.find("_batch0_"));
        if (batchName.find("_batch_") != std::string::npos)
        {
            originalTableName = batchName.substr(0, batchName.find("_batch_")) + "_original_" + originalTableName;
        }

        tableGroups[originalTableName].push_back(batchName);
    }

    // If only one original table involved, process as before
    if (tableGroups.size() == 1)
    {
        // Execute the query on each batch and combine results
        int index = 1;

        for (const auto &batchName : batchedTables)
        {
            std::cout << "batch " << index << '\n';
            index++;
            // Temporarily replace the table name in the query

            std::string originalTableName = batchName.substr(0, batchName.find("_batch0_"));
            if (batchName.find("_batch_") != std::string::npos)
            {
                originalTableName = batchName.substr(0, batchName.find("_batch_"));
            }

            std::regex pattern("_batch_([a-zA-Z0-9_]+)_batch0_");

            std::smatch matches;
            std::string tempAlias = "";
            if (std::regex_search(batchName, matches, pattern))
            {
                // std::cout << "Alias found: " << matches[1] << std::endl; // prints 'alias'
                tempAlias = matches[1];
            }
            else
            {
                // std::cout << "No alias found" << std::endl;
            }
            std::string batchQuery = replaceTableNameInQuery(query, originalTableName, batchName, tempAlias);

            // Create a new statement for this batch
            hsql::SQLParserResult result;
            hsql::SQLParser::parse(batchQuery, &result);

            if (result.isValid() && result.size() > 0)
            {
                const auto *batchStmt = static_cast<const hsql::SelectStatement *>(result.getStatement(0));

                if (batchStmt)
                {
                    // Execute the query on this batch
                    std::shared_ptr<Table> batchResult = plan_builder_->build(batchStmt, batchQuery)->execute();

                    // Combine with previous results
                    if (finalResult == nullptr)
                    {
                        finalResult = batchResult;
                    }
                    else
                    {
                        mergeResults(finalResult, batchResult);
                    }
                    batchResult.reset();
                }
            }
        }
    }
    else
    {
        // Process all combinations of batches from different original tables
        std::vector<std::map<std::string, std::string>> batchCombinations = generateBatchCombinations(tableGroups);
        int index = 1;
        for (const auto &combination : batchCombinations)
        {
            std::cout << "batch " << index << '\n';
            index++;
            // Create a query for this batch combination
            std::string batchQuery = query;
            for (const auto &[originalTable, batchTable] : combination)
            {
                std::string tempOriginalTaple = originalTable;
                if (tempOriginalTaple.find("_original_") != std::string::npos)
                {
                    tempOriginalTaple = tempOriginalTaple.substr(0, tempOriginalTaple.find("_original_"));
                }

                std::regex pattern("_batch_([a-zA-Z0-9_]+)_batch0_");

                std::smatch matches;
                std::string tempAlias = "";
                if (std::regex_search(batchTable, matches, pattern))
                {
                    // std::cout << "Alias found: " << matches[1] << std::endl; // prints 'alias'
                    tempAlias = matches[1];
                }
                else
                {
                    // std::cout << "No alias found" << std::endl;
                }

                batchQuery = replaceTableNameInQuery(batchQuery, tempOriginalTaple, batchTable, tempAlias);
            }

            // Create a new statement for this batch
            hsql::SQLParserResult result;
            hsql::SQLParser::parse(batchQuery, &result);

            if (result.isValid() && result.size() > 0)
            {
                const auto *batchStmt = static_cast<const hsql::SelectStatement *>(result.getStatement(0));

                if (batchStmt)
                {
                    // Execute the query on this batch combination
                    std::shared_ptr<Table> batchResult = plan_builder_->build(batchStmt, batchQuery)->execute();

                    // Combine with previous results
                    if (finalResult == nullptr)
                    {
                        finalResult = batchResult;
                    }
                    else
                    {
                        mergeResults(finalResult, batchResult);
                    }
                }
            }
        }
    }

    return finalResult;
}

// Helper function to generate all combinations of batches from different original tables
std::vector<std::map<std::string, std::string>> QueryExecutor::generateBatchCombinations(
    const std::map<std::string, std::vector<std::string>> &tableGroups)
{
    std::vector<std::map<std::string, std::string>> result;

    // We'll use a recursive approach to generate all combinations
    std::map<std::string, std::string> currentCombination;
    generateBatchCombinationsRecursive(tableGroups, currentCombination,
                                       tableGroups.begin(), result);

    return result;
}

void QueryExecutor::generateBatchCombinationsRecursive(
    const std::map<std::string, std::vector<std::string>> &tableGroups,
    std::map<std::string, std::string> currentCombination,
    std::map<std::string, std::vector<std::string>>::const_iterator currentGroup,
    std::vector<std::map<std::string, std::string>> &result)
{
    // Base case: we've gone through all groups
    if (currentGroup == tableGroups.end())
    {
        result.push_back(currentCombination);
        return;
    }

    // For the current group, try each batch
    const std::string &originalTable = currentGroup->first;
    const std::vector<std::string> &batches = currentGroup->second;

    auto nextGroup = currentGroup;
    ++nextGroup;

    for (const auto &batch : batches)
    {
        // Add this batch to the current combination
        std::map<std::string, std::string> newCombination = currentCombination;
        newCombination[originalTable] = batch;

        // Recurse to the next group
        generateBatchCombinationsRecursive(tableGroups, newCombination, nextGroup, result);
    }
}

std::string QueryExecutor::replaceTableNameInQuery(const std::string &query,
                                                   const std::string &oldName,
                                                   const std::string &newName,
                                                   const std::string &oldAlias)
{
    std::string modifiedQuery = query;

    size_t pos = 0;
    while ((pos = modifiedQuery.find(oldName, pos)) != std::string::npos)
    {
        // Check if the table name is followed by " AS " and then the alias
        size_t asPos = modifiedQuery.find(" AS ", pos + oldName.length());

        if (asPos != std::string::npos)
        {
            size_t aliasPos = modifiedQuery.find(oldAlias, asPos + 4); // +4 to skip " AS "

            if (aliasPos != std::string::npos)
            {
                // Make sure alias is a standalone part (not part of another identifier)
                bool isStandalone = true;
                if (aliasPos > 0 && (isalnum(modifiedQuery[aliasPos - 1]) || modifiedQuery[aliasPos - 1] == '_'))
                {
                    isStandalone = false;
                }
                if (aliasPos + oldAlias.length() < modifiedQuery.length() &&
                    (isalnum(modifiedQuery[aliasPos + oldAlias.length()]) || modifiedQuery[aliasPos + oldAlias.length()] == '_'))
                {
                    isStandalone = false;
                }

                if (isStandalone)
                {
                    // Replace the table name only if " AS <alias>" is found
                    modifiedQuery.replace(pos, oldName.length(), newName);
                    pos += newName.length(); // Move position forward to avoid multiple replacements
                }
                else
                {
                    pos += oldName.length(); // If alias isn't standalone, skip
                }
            }
            else
            {
                pos += oldName.length(); // Skip if alias isn't found after " AS "
            }
        }
        else
        {
            pos += oldName.length(); // Skip if " AS " isn't found after the table name
        }
    }
    pos = 0;
    while ((pos = modifiedQuery.find(oldName, pos)) != std::string::npos)
    {
        size_t asPos = modifiedQuery.find(" ", pos + oldName.length()); // Find the first space after the table name

        if (asPos != std::string::npos)
        {
            // Check if the next part is a valid alias (it should be a valid identifier)
            size_t aliasPos = asPos + 1; // Skip the space after the table name

            // Find the end of the alias (it should either be followed by a space or end of string)
            size_t aliasEndPos = modifiedQuery.find_first_of(",", aliasPos);
            if (aliasEndPos == std::string::npos)
            {
                aliasEndPos = modifiedQuery.find_first_of(" ", aliasPos);
            }
            if (aliasEndPos == std::string::npos)
            {
                aliasEndPos = modifiedQuery.length(); // Alias goes till the end of the string
            }

            // Extract the alias
            std::string foundAlias = modifiedQuery.substr(aliasPos, aliasEndPos - aliasPos);

            // Check if the found alias matches the old alias
            if (foundAlias == oldAlias)
            {
                // Ensure the alias is standalone (not part of another identifier)
                bool isStandalone = true;
                if (aliasPos > 0 && (isalnum(modifiedQuery[aliasPos - 1]) || modifiedQuery[aliasPos - 1] == '_'))
                {
                    isStandalone = false; // Alias isn't standalone if preceded by alphanumeric or '_'
                }
                if (aliasEndPos < modifiedQuery.length() &&
                    (isalnum(modifiedQuery[aliasEndPos]) || modifiedQuery[aliasEndPos] == '_'))
                {
                    isStandalone = false; // Alias isn't standalone if followed by alphanumeric or '_'
                }

                if (isStandalone)
                {
                    // Replace the table name only if alias is found and is valid
                    modifiedQuery.replace(pos, oldName.length(), newName);
                    pos += newName.length(); // Move position forward to avoid multiple replacements
                }
                else
                {
                    pos += oldName.length(); // If alias isn't standalone, skip
                }
            }
            else
            {
                pos += oldName.length(); // Skip if the alias doesn't match
            }
        }
        else
        {
            pos += oldName.length(); // Skip if no space is found after table name
        }
    }
    return modifiedQuery;
}

void QueryExecutor::cleanupBatchTables()
{
    // Collect table names first to avoid modifying the map while iterating
    std::vector<std::string> allTableNames = storage_->getTableNames();
    std::unordered_map<std::string, std::string> largeToOriginalMap;

    // First pass: identify large tables and their original name
    for (const auto &tableName : allTableNames)
    {
        if (tableName.find("_large") != std::string::npos)
        {
            std::string originalName = tableName.substr(0, tableName.find("_large"));
            largeToOriginalMap[tableName] = originalName;
        }
    }

    // Second pass: remove all batch tables
    for (const auto &tableName : allTableNames)
    {
        if (tableName.find("_batch0_") != std::string::npos)
        {
            storage_->tables.erase(tableName);
        }
    }

    // Third pass: rename large tables back to original names
    for (const auto &[largeTableName, originalName] : largeToOriginalMap)
    {
        if (storage_->tableExists(largeTableName))
        {
            // Get the large table and rename it back to the original name
            // We need to create a new table with the original name
            auto &largeTable = storage_->getTable(largeTableName);
            storage_->tables[originalName] = std::make_unique<Table>(
                originalName,
                largeTable.getHeaders(),
                largeTable.getData(),
                largeTable.getColumnTypes());

            // Remove the large table
            storage_->tables.erase(largeTableName);
        }
    }
}

void QueryExecutor::mergeResults(std::shared_ptr<Table> table1,
                                 std::shared_ptr<Table> table2)
{
    // Ensure schemas match
    const std::vector<std::string> &headers1 = table1->getHeaders();
    const auto &types1 = table1->getColumnTypes();

    // Access mutable data from table1
    const auto &data2 = table2->getData();

    for (int row = 0; row < table2->getSize(); row++)
    {
        std::vector<unionV> tempUnionVector;

        for (const auto &header : headers1)
        {
            const auto &col2 = data2.at(header);
            ColumnType type = types1.at(header);
            auto tempUnion = copyUnionValue(col2[row], type);
            tempUnionVector.push_back(tempUnion);
        }
        table1->addRow(tempUnionVector);
    }
}

// Update validateSelectStatement
void QueryExecutor::validateSelectStatement(const hsql::SelectStatement *stmt)
{
    if (!stmt->fromTable)
    {
        throw SemanticError("Missing FROM clause");
    }

    // Validate projection list
    // for (const auto *expr : *(stmt->selectList))
    // {
    //     if (expr->type == hsql::kExprFunctionRef)
    //     {
    //         throw SemanticError("Aggregate functions not yet supported");
    //     }
    // }
}