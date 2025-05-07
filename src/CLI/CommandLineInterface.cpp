#include "../../include/CLI/CommandLineInterface.hpp"
#include "../../include/DataHandling/CSVProcessor.hpp"
#include <iostream>
#include <sstream>
#include <chrono>

using namespace std::chrono;

CommandLineInterface::CommandLineInterface()
    : storageManager(std::make_unique<StorageManager>()) {}

void CommandLineInterface::run()
{
    std::cout << "SQL-like Query Processor\nType 'help' for commands\n";

    std::string input;
    while (true)
    {
        std::cout << "> ";
        std::getline(std::cin, input);

        if (input.empty())
            continue;
        if (input == "exit" || input == "quit")
            break;

        try
        {
            if (input == "help")
            {
                displayHelp();
            }
            else if (input == "show tables")
            {
                handleShowTablesCommand();
            }
            else if (input.rfind("load ", 0) == 0)
            {
                std::istringstream iss(input);
                std::vector<std::string> args;
                std::string cmd, arg;
                iss >> cmd;
                while (iss >> arg)
                    args.push_back(arg);
                handleLoadCommand(args);
            }
            else if (input.rfind("set mode ", 0) == 0)
            {
                std::string mode = input.substr(9); // "set mode " is 9 characters
                if (mode == "CPU")
                {
                    QueryExecutor::setExecutionMode(ExecutionMode::CPU);
                    std::cout << "\n=== CPU Info ===" << std::endl;
                    system("cat /proc/cpuinfo | grep 'model name' | uniq");
                }
                else if (mode == "GPU")
                {
                    QueryExecutor::setExecutionMode(ExecutionMode::GPU);
                    std::cout << "\n=== GPU Info ===" << std::endl;
                    system("nvidia-smi --query-gpu=name --format=csv,noheader");
                }
                else
                {
                    std::cout << "Invalid mode. Use 'CPU' or 'GPU'.\n";
                }
            }

            else
            {
                processQuery(input);
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error: " << e.what() << "\n";
        }
    }
}

void CommandLineInterface::cleanupBatchTables()
{
    // Collect table names first to avoid modifying the map while iterating
    std::vector<std::string> allTableNames = storageManager->getTableNames();
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

    for (const auto &tableName : allTableNames)
    {
        if (tableName.find("sub_query") != std::string::npos)
        {
            std::string originalName = tableName.substr(0, tableName.find("_large"));
            largeToOriginalMap[tableName] = originalName;
        }
    }

    // Second pass: remove all batch tables
    for (const auto &tableName : allTableNames)
    {
        if (tableName.find("_batch_") != std::string::npos)
        {
            storageManager->tables.erase(tableName);
        }
    }

    // Third pass: rename large tables back to original names
    for (const auto &[largeTableName, originalName] : largeToOriginalMap)
    {
        if (storageManager->tableExists(largeTableName))
        {
            // Get the large table and rename it back to the original name
            // We need to create a new table with the original name
            auto &largeTable = storageManager->getTable(largeTableName);
            storageManager->tables[originalName] = std::make_unique<Table>(
                originalName,
                largeTable.getHeaders(),
                largeTable.getData(),
                largeTable.getColumnTypes());

            // Remove the large table
            storageManager->tables.erase(largeTableName);
        }
    }
}
void CommandLineInterface::processQuery(const std::string &query)
{
    try
    {
        auto start = high_resolution_clock::now();

        QueryExecutor executor(storageManager);
        std::shared_ptr<Table> result = executor.execute(query);

        if (result)
        {
            const auto &columns = result->getHeaders();
            const auto &columnData = result->getData(); // Now returns unordered_map<string, vector<string>>
            const int totalRows = result->getSize();
            const auto columnTypes = result->getColumnTypes();

            // Display column headers
            for (size_t i = 0; i < columns.size(); ++i)
            {
                std::cout << columns[i];
                if (i < columns.size() - 1)
                    std::cout << " | ";
            }
            std::cout << "\n";

            // Display separator line
            for (size_t i = 0; i < columns.size(); ++i)
            {
                std::cout << std::string(columns[i].length(), '-');
                if (i < columns.size() - 1)
                    std::cout << "-+-";
            }
            std::cout << "\n";

            // Display first 10 rows
            const size_t numRowsToShow = std::min<size_t>(totalRows, 10);
            for (size_t rowIdx = 0; rowIdx < numRowsToShow; ++rowIdx)
            {
                // For each row, go through all columns and get the value at that row index
                for (size_t colIdx = 0; colIdx < columns.size(); ++colIdx)
                {
                    const auto &colName = columns[colIdx];
                    switch (columnTypes.at(columns[colIdx]))
                    {
                    case ColumnType::INTEGER:
                        std::cout << columnData.at(colName)[rowIdx].i;
                        break;

                    case ColumnType::STRING:
                        std::cout << *(columnData.at(colName)[rowIdx].s);
                        break;

                    case ColumnType::DOUBLE:
                        std::cout << columnData.at(colName)[rowIdx].d;
                        break;

                    default:
                        throw std::runtime_error("Unsupported column type");
                    }
                    if (colIdx < columns.size() - 1)
                        std::cout << " | ";
                }
                std::cout << "\n";
            }

            // Show truncation message if needed
            if (totalRows > 10)
            {
                std::cout << "...\n";
                std::cout << "(Showing first 10 of " << totalRows << " rows)\n";
            }

            std::cout << totalRows << " rows returned\n";

            // Save full results to CSV
            std::string outputPath = "../../data/output/query_output.csv";
            CSVProcessor::saveCSV(outputPath, result->getHeaders(), columnData, columnTypes); // CSVProcessor needs to be updated too
            std::cout << "Saved output to '" << outputPath << "'\n";

            auto end = high_resolution_clock::now();

            auto duration = duration_cast<milliseconds>(end - start);

            std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
            cleanupBatchTables();
            result.reset();
        }
        else
        {
            std::cout << "Query executed successfully but returned no results\n";
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Query execution error: " << e.what() << "\n";
    }
}

void CommandLineInterface::handleLoadCommand(const std::vector<std::string> &args)
{
    if (args.size() != 2)
    {
        std::cout << ("Usage: load <table_name> <filepath>");
        return;
    }

    const std::string &tableName = args[0];
    const std::string &filepath = args[1];

    auto start = high_resolution_clock::now();

    storageManager->loadTable(tableName, filepath);
    std::cout << "Loaded table '" << tableName << "' from " << filepath << "\n";
    auto end = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end - start);

    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
}

void CommandLineInterface::handleShowTablesCommand()
{
    // TODO: Implement table listing
    auto tables = storageManager->listTables();
    std::cout << "Tables loaded:\n";
    if (tables.empty())
    {
        std::cout << "  (none)\n";
    }
    else
    {
        for (const auto &table : tables)
        {
            std::cout << "  " << table << "\n";
        }
    }
}

void CommandLineInterface::displayHelp()
{
    std::cout << "Available commands:\n"
              << "  load <table_name> <filepath> - Load a CSV file as a table\n"
              << "  set mode <GPU>               - List all loaded tables\n"
              << "  <SQL query>                  - Execute a query\n"
              << "  help                         - Show this help\n"
              << "  exit/quit                    - Exit the program\n";
}