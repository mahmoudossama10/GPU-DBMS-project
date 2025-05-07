#include "../../include/QueryProcessing/QueryExecutor.hpp"
#include "../../include/QueryProcessing/PlanBuilder.hpp"
#include "Utilities/ErrorHandling.hpp"
#include "Utilities/UnionUtils.hpp"

#include <iostream>
#include <hsql/util/sqlhelper.h>
#include <regex>
#include <memory>
#include <unordered_set>

QueryExecutor::QueryExecutor(std::shared_ptr<StorageManager> storage)
    : storage_(storage),
      plan_builder_(std::make_unique<PlanBuilder>(storage)) {}

std::string QueryExecutor::removeAggregatesFromQuery(const std::string &query)
{
    // First, find the SELECT and FROM positions to identify our working area
    std::string upperQuery = query;
    std::transform(upperQuery.begin(), upperQuery.end(), upperQuery.begin(),
                   [](unsigned char c)
                   { return std::toupper(c); });

    size_t selectPos = upperQuery.find("SELECT");
    size_t fromPos = upperQuery.find("FROM");

    if (selectPos == std::string::npos || fromPos == std::string::npos || selectPos >= fromPos)
    {
        // Invalid SQL query format
        return query;
    }

    // Extract the selection part
    std::string selectionPart = query.substr(selectPos + 6, fromPos - (selectPos + 6));
    std::string restOfQuery = query.substr(fromPos);

    // Split the selection part by commas
    std::vector<std::string> columns;
    size_t start = 0;
    bool insideParentheses = false;

    for (size_t i = 0; i < selectionPart.length(); i++)
    {
        if (selectionPart[i] == '(')
        {
            insideParentheses = true;
        }
        else if (selectionPart[i] == ')')
        {
            insideParentheses = false;
        }
        else if (selectionPart[i] == ',' && !insideParentheses)
        {
            columns.push_back(selectionPart.substr(start, i - start));
            start = i + 1;
        }
    }

    // Add the last column
    columns.push_back(selectionPart.substr(start));

    // Process each column to remove aggregates
    std::vector<std::string> filteredColumns;
    std::regex aggregatePattern("\\s*(COUNT|SUM|AVG|MIN|MAX)\\s*\\([^)]*\\)\\s*(?:AS\\s+([^,\\s]+))?",
                                std::regex_constants::icase);

    for (const auto &column : columns)
    {
        std::smatch matches;
        if (!std::regex_search(column, matches, aggregatePattern))
        {
            // Only keep columns that are not aggregates
            filteredColumns.push_back(column);
        }
        // We completely skip aggregates now, not preserving aliases
    }

    // Rebuild the query
    std::string result = "SELECT ";

    // If no columns remain after removing aggregates, add a * wildcard
    if (filteredColumns.empty())
    {
        result += "* ";
    }
    else
    {
        for (size_t i = 0; i < filteredColumns.size(); i++)
        {
            if (i > 0)
            {
                result += ", ";
            }
            result += filteredColumns[i];
        }
        result += " ";
    }

    result += restOfQuery;

    return result;
}

SubqueryInfo QueryExecutor::extractSubquery(const std::string &query)
{
    SubqueryInfo result;
    result.alias = "";
    result.original_query = query;
    result.after_from = false;
    result.after_where_in = false;

    // Pattern to match subqueries - looks for pattern ( SELECT ... ) [AS] alias
    // Using regex with balanced parentheses is tricky, so we'll use a different approach

    // Find the position of the first opening parenthesis that is followed by "SELECT" or "select"
    size_t openPos = std::string::npos;
    for (size_t i = 0; i < query.length(); ++i)
    {
        if (query[i] == '(')
        {
            // Check if "SELECT" or "select" follows, ignoring whitespace
            size_t j = i + 1;
            while (j < query.length() && std::isspace(query[j]))
                ++j;

            if (j + 6 <= query.length() &&
                (query.substr(j, 6) == "SELECT" || query.substr(j, 6) == "select"))
            {
                openPos = i;

                // Check if the subquery comes after FROM keyword
                std::string beforeSubquery = query.substr(0, i);
                std::regex fromPattern(R"(FROM\s+$|FROM\s+.*?[^\w])");
                if (std::regex_search(beforeSubquery, fromPattern))
                {
                    result.after_from = true;
                }

                // Check if the subquery comes after IN keyword following WHERE
                std::regex whereInPattern(R"(\bWHERE\b)");
                if (std::regex_search(beforeSubquery, whereInPattern))
                {
                    result.after_where_in = true;
                }

                break;
            }
        }
    }

    if (openPos == std::string::npos)
    {
        // No subquery found
        result.modified_query = query;
        return result;
    }

    // Find the matching closing parenthesis
    size_t closePos = openPos;
    int level = 0;

    for (size_t i = openPos; i < query.length(); ++i)
    {
        if (query[i] == '(')
        {
            level++;
        }
        else if (query[i] == ')')
        {
            level--;
            if (level == 0)
            {
                closePos = i;
                break;
            }
        }
    }

    if (level != 0)
    {
        // Unbalanced parentheses
        result.modified_query = query;
        return result;
    }

    // Extract the subquery (without the parentheses)
    result.subquery = query.substr(openPos + 1, closePos - openPos - 1);

    // Look for an alias after the closing parenthesis
    std::regex aliasRegex(R"(\)\s+(?:AS\s+|as\s+)?([a-zA-Z][a-zA-Z0-9_]*))");
    std::smatch aliasMatch;

    std::string afterClose = query.substr(closePos);
    if (std::regex_search(afterClose, aliasMatch, aliasRegex))
    {
        result.alias = aliasMatch[1].str();
    }

    // Replace the subquery WITH ITS BRACKETS with just "sub_query" in the original query
    result.modified_query = query.substr(0, openPos) + "sub_query" + query.substr(closePos + 1);

    return result;
}

std::shared_ptr<Table> QueryExecutor::execute(const std::string &query)
{

    // hsql::printStatementInfo(result.getStatement(0));

    auto subQuery = extractSubquery(query);

    if (subQuery.subquery == "")
    {
        hsql::SQLParserResult result;
        hsql::SQLParser::parse(query, &result);
        if (!result.isValid() || result.size() == 0)
        {
            throw std::runtime_error("Failed to parse SQL query or no valid statement found.");
        }
        const auto *stmt = result.getStatement(0);
        if (!stmt || stmt->type() != hsql::kStmtSelect)
        {
            throw std::runtime_error("Parsed statement is not a valid SELECT statement.");
        }
        return execute(static_cast<const hsql::SelectStatement *>(stmt), query);
    }
    else
    {
        hsql::SQLParserResult result;
        hsql::SQLParser::parse(subQuery.subquery, &result);
        const auto *stmt = result.getStatement(0);
        auto resultTable = execute(static_cast<const hsql::SelectStatement *>(stmt), subQuery.subquery);
        resultTable->setAlias(subQuery.alias);
        // Step 1: Preview the stripped headers
        std::vector<std::string> strippedHeaders;
        strippedHeaders.reserve(resultTable->headers.size());

        for (const auto &header : resultTable->headers)
        {
            size_t lastDot = header.rfind('.');
            if (lastDot != std::string::npos && lastDot + 1 < header.size())
            {
                strippedHeaders.push_back(header.substr(lastDot + 1));
            }
            else
            {
                strippedHeaders.push_back(header); // No change
            }
        }

        // Step 2: Check for duplicates
        std::unordered_set<std::string> seen;
        bool hasDuplicates = false;
        for (const auto &h : strippedHeaders)
        {
            if (!seen.insert(h).second)
            {
                hasDuplicates = true;
                break;
            }
        }

        // Step 3: Apply only if no duplicates
        if (!hasDuplicates)
        {
            resultTable->headers = strippedHeaders;
        }

        if (subQuery.after_where_in)
        {
            auto aggValue = resultTable->getRow(0)[0];
            auto aggType = resultTable->getColumnType(resultTable->headers[0]);

            // Create a deep copy of the union value if needed
            auto stringValue = UnionUtils::valueToString(aggValue, aggType);

            std::string toReplace = "sub_query";

            size_t pos = subQuery.modified_query.find(toReplace);
            if (pos != std::string::npos)
            {
                subQuery.modified_query.replace(pos, toReplace.length(), stringValue);
            }
        }
        else
        {
            std::string oldName = resultTable->getName();
            std::string newName = "sub_query";

            auto it = storage_->tables.find(oldName);
            if (it != storage_->tables.end())
            {
                storage_->renameTable(resultTable->getName(), "sub_query");
                storage_->tables.at("sub_query")->setAlias(subQuery.alias);
            }
            else
            {
                resultTable->tableName = "sub_query";
                storage_->addTable(resultTable);
                storage_->tables.at("sub_query")->setAlias(subQuery.alias);
            }
        }

        hsql::SQLParserResult result2;
        hsql::SQLParser::parse(subQuery.modified_query, &result2);
        const auto *stmt2 = result2.getStatement(0);

        std::string outputPath = "../../data/input/sub_query.csv";
        CSVProcessor::saveCSV(outputPath, resultTable->getHeaders(), resultTable->getData(), resultTable->getColumnTypes()); // CSVProcessor needs to be updated too
        std::cout << "Saved output to '" << outputPath << "'\n";

        cleanupBatchTables();
        return execute(static_cast<const hsql::SelectStatement *>(stmt2), subQuery.modified_query);
    }
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
    size_t BATCH_SIZE = 500;
    if (allTableNamesOriginal.size() == 1)
    {
        BATCH_SIZE = storage_->getTable(allTableNamesOriginal[0]).getSize();
    }

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

    auto newQuery = removeAggregatesFromQuery(query);

    std::shared_ptr<Table> result = processBatchedQuery(stmt, newQuery);

    /*
    agg

    call aggregate with hsql select list

    project

    check if their is other thing than agg fucntions do the project and then add a new columns with aggregate values of the sam table size and then addthese new columns
    to the table



    order by

    call order by

    */

    hsql::SQLParserResult resultWithoutAgg;
    hsql::SQLParser::parse(newQuery, &resultWithoutAgg);
    const auto *stmtWithoutAgg = static_cast<const hsql::SelectStatement *>(resultWithoutAgg.getStatement(0));

    std::shared_ptr<Table> aggResult;
    if (plan_builder_->hasAggregates(*(stmt->selectList)))
    {
        if (PlanBuilder::execution_mode_ == ExecutionMode::CPU)
        {
            aggResult = plan_builder_->buildCPUAggregatePlan(result, *(stmt->selectList));
        }
        else
        {
            aggResult = plan_builder_->buildGPUAggregatePlan(result, *(stmt->selectList));
        }
    }

    if (stmt->order && !stmt->order->empty())
    {
        if (PlanBuilder::execution_mode_ == ExecutionMode::CPU)
        {
            result = plan_builder_->buildOrderByPlan(result, *stmt->order);
        }
        else
        {
            result = plan_builder_->buildGPUOrderByPlan(result, *stmt->order);
        }
    }

    if (!plan_builder_->isSelectAll(stmtWithoutAgg->selectList) && plan_builder_->selectListNeedsProjection(*(stmtWithoutAgg->selectList)))
    {
        result = plan_builder_->buildProjectPlan(result, *(stmtWithoutAgg->selectList));
    }

    if (plan_builder_->hasAggregates(*(stmt->selectList)) && plan_builder_->hasOtherSelectNotAggregates(*(stmt->selectList)))
    {
        // Prepare new data structures
        std::unordered_map<std::string, std::vector<unionV>> newColumnData;
        std::unordered_map<std::string, ColumnType> newColumnTypes;
        std::vector<std::string> newHeaders;

        size_t targetSize = result->getSize();

        for (size_t i = 0; i < result->getHeaders().size(); i++)
        {
            std::string columnName = result->getHeaders()[i];

            // Store column data and metadata
            newColumnData[columnName] = std::move(result->columnData.at(columnName));
            newColumnTypes[columnName] = result->getColumnType(columnName);
            newHeaders.push_back(columnName);
        }

        for (size_t i = 0; i < aggResult->getHeaders().size(); i++)
        {
            std::string columnName = aggResult->getHeaders()[i];

            // Extract the first value from the aggregated column
            auto firstValue = aggResult->columnData.at(columnName)[0];

            // Broadcast the value
            std::vector<unionV> newColumn(targetSize, firstValue);

            // Store column data and metadata
            newColumnData[columnName] = std::move(newColumn);
            newColumnTypes[columnName] = aggResult->getColumnType(columnName);
            newHeaders.push_back(columnName);
        }

        // Create the new table using the accumulated data
        result = std::make_shared<Table>("final_result_large", newHeaders, newColumnData, newColumnTypes);
    }
    else if (plan_builder_->hasAggregates(*(stmt->selectList)))
    {
        result = aggResult;
    }

    // result = plan_builder_->buildOrderByPlan(result, *stmt->order);
    // Cleanup batch tables and restore original tables
    cleanupBatchTables();
    return result;
}

// Helper function to copy union values based on type (similar to the one in your Table class)
unionV QueryExecutor::copyUnionValue(const unionV &value, ColumnType type)
{
    unionV copy = {};
    copy.i = new TheInteger();
    copy.d = new TheDouble();

    switch (type)
    {
    case ColumnType::STRING:
        copy.s = (value.s != nullptr) ? new std::string(*value.s) : nullptr;
        break;
    case ColumnType::INTEGER:
        copy.i->value = value.i->value;
        copy.i->is_null = value.i->is_null;
        break;
    case ColumnType::DOUBLE:
        copy.d->value = value.d->value;
        copy.d->is_null = value.d->is_null;
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
                    finalResult = batchResult;
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
                    finalResult = batchResult;
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

    // If oldAlias is empty, simply replace all occurrences of oldName with newName
    // making sure they are standalone identifiers
    if (oldAlias.empty())
    {
        pos = 0;
        while ((pos = modifiedQuery.find(oldName, pos)) != std::string::npos)
        {
            // Check if this is a standalone occurrence of the table name
            bool isStandalone = true;

            // Check if preceded by non-identifier character
            if (pos > 0 && (isalnum(modifiedQuery[pos - 1]) || modifiedQuery[pos - 1] == '_'))
            {
                isStandalone = false;
            }

            // Check if followed by non-identifier character or end of string
            size_t endPos = pos + oldName.length();
            if (endPos < modifiedQuery.length() &&
                (isalnum(modifiedQuery[endPos]) || modifiedQuery[endPos] == '_'))
            {
                isStandalone = false;
            }

            if (isStandalone)
            {
                // Replace the table name
                modifiedQuery.replace(pos, oldName.length(), newName);
                pos += newName.length(); // Move position forward to avoid multiple replacements
            }
            else
            {
                pos += 1; // Skip just one character to find potential matches later
            }
        }

        return modifiedQuery;
    }

    // First pass: Find table name with explicit AS alias
    pos = 0;
    while ((pos = modifiedQuery.find(oldName, pos)) != std::string::npos)
    {
        // Check if this is a standalone occurrence of the table name
        bool isStandalone = true;

        // Check if preceded by non-identifier character
        if (pos > 0 && (isalnum(modifiedQuery[pos - 1]) || modifiedQuery[pos - 1] == '_'))
        {
            isStandalone = false;
        }

        // Check if followed by non-identifier character or end of string
        size_t endPos = pos + oldName.length();
        if (endPos < modifiedQuery.length() &&
            (isalnum(modifiedQuery[endPos]) || modifiedQuery[endPos] == '_'))
        {
            isStandalone = false;
        }

        if (!isStandalone)
        {
            pos += 1; // Skip just one character to find potential matches later
            continue;
        }

        // Check if the table name is followed by " AS " and then the alias
        size_t asPos = modifiedQuery.find(" AS ", pos + oldName.length());

        if (asPos != std::string::npos && asPos == pos + oldName.length())
        {
            size_t aliasPos = asPos + 4; // +4 to skip " AS "
            size_t aliasEndPos = aliasPos + oldAlias.length();

            // Check if the alias matches and is standalone
            if (modifiedQuery.substr(aliasPos, oldAlias.length()) == oldAlias &&
                (aliasEndPos >= modifiedQuery.length() ||
                 !(isalnum(modifiedQuery[aliasEndPos]) || modifiedQuery[aliasEndPos] == '_')))
            {
                // Replace the table name
                modifiedQuery.replace(pos, oldName.length(), newName);
                pos += newName.length(); // Move position forward
            }
            else
            {
                pos += oldName.length(); // Skip this occurrence
            }
        }
        else
        {
            pos += oldName.length(); // Skip if " AS " isn't found immediately after the table name
        }
    }

    // Second pass: Find table name with implicit alias (without AS keyword)
    pos = 0;
    while ((pos = modifiedQuery.find(oldName, pos)) != std::string::npos)
    {
        // Check if this is a standalone occurrence of the table name
        bool isStandalone = true;

        // Check if preceded by non-identifier character
        if (pos > 0 && (isalnum(modifiedQuery[pos - 1]) || modifiedQuery[pos - 1] == '_'))
        {
            isStandalone = false;
        }

        // Check if followed by non-identifier character or end of string
        size_t endPos = pos + oldName.length();
        if (endPos < modifiedQuery.length() &&
            (isalnum(modifiedQuery[endPos]) || modifiedQuery[endPos] == '_'))
        {
            isStandalone = false;
        }

        if (!isStandalone)
        {
            pos += 1; // Skip just one character to find potential matches later
            continue;
        }

        // Check if the table name is followed by space and then directly by the alias
        if (endPos < modifiedQuery.length() && modifiedQuery[endPos] == ' ')
        {
            size_t aliasPos = endPos + 1; // Skip the space after the table name

            // Check if this space is followed by the alias
            if (aliasPos + oldAlias.length() <= modifiedQuery.length() &&
                modifiedQuery.substr(aliasPos, oldAlias.length()) == oldAlias)
            {
                // Check if the alias is standalone
                size_t aliasEndPos = aliasPos + oldAlias.length();
                if (aliasEndPos >= modifiedQuery.length() ||
                    !(isalnum(modifiedQuery[aliasEndPos]) || modifiedQuery[aliasEndPos] == '_'))
                {
                    // Replace the table name
                    modifiedQuery.replace(pos, oldName.length(), newName);
                    pos += newName.length(); // Move position forward
                    continue;
                }
            }
        }

        // Check if we have a comma-separated list and need to look for an alias
        size_t commaPos = modifiedQuery.find(",", pos + oldName.length());
        if (commaPos != std::string::npos &&
            modifiedQuery.find_first_not_of(" \t", pos + oldName.length()) == commaPos)
        {
            // Found a comma right after the table name (with optional spaces)
            // This is a standalone table reference without an alias
            modifiedQuery.replace(pos, oldName.length(), newName);
            pos += newName.length();
        }
        else
        {
            // Check for other clause keywords
            std::vector<std::string> clauses = {" WHERE ", " FROM ", " JOIN ", " ON ", " GROUP ", " HAVING ", " ORDER ", " LIMIT "};
            bool foundClause = false;

            for (const auto &clause : clauses)
            {
                size_t clausePos = modifiedQuery.find(clause, pos + oldName.length());
                if (clausePos != std::string::npos &&
                    modifiedQuery.find_first_not_of(" \t", pos + oldName.length()) == clausePos)
                {
                    // Found a SQL clause right after the table name
                    modifiedQuery.replace(pos, oldName.length(), newName);
                    pos += newName.length();
                    foundClause = true;
                    break;
                }
            }

            if (!foundClause)
            {
                pos += oldName.length(); // Skip this occurrence
            }
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