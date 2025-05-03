#include "../../include/CLI/CommandLineInterface.hpp"
#include "../../include/DataHandling/CSVProcessor.hpp"
#include <iostream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <cctype>

using namespace std::chrono;

CommandLineInterface::CommandLineInterface()
    : storageManager(std::make_unique<StorageManager>()), historyIndex(0)
{
    linenoiseSetMultiLine(1);
    linenoiseHistorySetMaxLen(100);
}

CommandLineInterface::~CommandLineInterface()
{
    linenoiseHistorySave("history.txt");
}

std::string CommandLineInterface::getInput()
{
    char *line = linenoise("> ");
    if (line == nullptr)
    {
        return "quit";
    }
    std::string input = line;
    linenoiseHistoryAdd(line);
    free(line);
    return input;
}

void CommandLineInterface::run()
{
    std::cout << "SQL-like Query Processor\nType 'help' for commands\n";

    while (true)
    {
        std::string input = getInput();

        if (input.empty())
        {
            continue;
        }

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
                std::istringstream iss(input); // Use original input for args
                std::vector<std::string> args;
                std::string cmd, arg;
                iss >> cmd;
                while (iss >> arg)
                {
                    args.push_back(arg);
                }
                handleLoadCommand(args);
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
            const auto &columnData = result->getData();
            const int totalRows = result->getSize();
            const auto &columnTypes = result->getColumnTypes();

            // Display column headers
            for (size_t i = 0; i < columns.size(); ++i)
            {
                std::cout << columns[i];
                if (i < columns.size() - 1)
                {
                    std::cout << " | ";
                }
            }
            std::cout << "\n";

            // Display separator line
            for (size_t i = 0; i < columns.size(); ++i)
            {
                std::cout << std::string(columns[i].length(), '-');
                if (i < columns.size() - 1)
                {
                    std::cout << "-+-";
                }
            }
            std::cout << "\n";

            // Display first 10 rows
            const size_t numRowsToShow = std::min<size_t>(totalRows, 10);
            for (size_t rowIdx = 0; rowIdx < numRowsToShow; ++rowIdx)
            {
                for (size_t colIdx = 0; colIdx < columns.size(); ++colIdx)
                {
                    const auto &colName = columns[colIdx];
                    switch (columnTypes.at(colName))
                    {
                    case ColumnType::INTEGER:
                        std::cout << columnData.at(colName)[rowIdx].i;
                        break;
                    case ColumnType::STRING:
                        std::cout << *(columnData.at(colName)[rowIdx].s);
                        break;
                    default:
                        throw std::runtime_error("Unsupported column type");
                    }
                    if (colIdx < columns.size() - 1)
                    {
                        std::cout << " | ";
                    }
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
            std::string outputPath = "./data/output/query_output.csv";
            CSVProcessor::saveCSV(outputPath, result->getHeaders(), columnData, columnTypes);
            std::cout << "Saved output to '" << outputPath << "'\n";

            auto end = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(end - start);
            std::cout << "Execution time: " << duration.count() << " ms\n";
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
        std::cerr << "Usage: load <table_name> <filepath>\n";
        return;
    }

    const std::string &tableName = args[0];
    const std::string &filepath = args[1];

    try
    {
        storageManager->loadTable(tableName, filepath);
        std::cout << "Loaded table '" << tableName << "' from " << filepath << "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error loading table: " << e.what() << "\n";
    }
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
              << "  show tables                  - List all loaded tables\n"
              << "  <SQL query>                  - Execute a SQL-like query\n"
              << "  help                         - Show this help\n"
              << "  exit/quit                    - Exit the program\n";
}