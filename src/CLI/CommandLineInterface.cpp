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
            const auto &rows = result->getData();

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
            const size_t numRowsToShow = std::min<size_t>(rows.size(), 10);
            for (size_t rowIdx = 0; rowIdx < numRowsToShow; ++rowIdx)
            {
                const auto &row = rows[rowIdx];
                for (size_t i = 0; i < row.size(); ++i)
                {
                    std::cout << row[i];
                    if (i < row.size() - 1)
                        std::cout << " | ";
                }
                std::cout << "\n";
            }

            // Show truncation message if needed
            if (rows.size() > 10)
            {
                std::cout << "...\n";
                std::cout << "(Showing first 10 of " << rows.size() << " rows)\n";
            }

            std::cout << rows.size() << " rows returned\n";

            // Save full results to CSV
            std::string outputPath = "./data/output/query_output.csv";
            CSVProcessor::saveCSV(outputPath, result->getHeaders(), result->getData());
            std::cout << "Saved output to '" << outputPath << "'\n";

            auto end = high_resolution_clock::now();

            auto duration = duration_cast<milliseconds>(end - start); // You can use microseconds, seconds, etc.

            std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
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

    storageManager->loadTable(tableName, filepath);
    std::cout << "Loaded table '" << tableName << "' from " << filepath << "\n";
}

void CommandLineInterface::handleShowTablesCommand()
{
    // TODO: Implement table listing
    std::cout << "Tables loaded: \n";
}

void CommandLineInterface::displayHelp()
{
    std::cout << "Available commands:\n"
              << "  load <table_name> <filepath> - Load a CSV file as a table\n"
              << "  show tables                  - List all loaded tables\n"
              << "  <SQL query>                  - Execute a query\n"
              << "  help                         - Show this help\n"
              << "  exit/quit                    - Exit the program\n";
}