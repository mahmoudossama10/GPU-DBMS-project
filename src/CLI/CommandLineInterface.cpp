#include "../../include/CLI/CommandLineInterface.hpp"
#include "../../include/DataHandling/CSVProcessor.hpp"
#include <iostream>
#include <fstream>
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
            else if (input == "test")
            {
                handleTestCommand();
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
                {
                    args.push_back(arg);
                }
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
        // auto start = high_resolution_clock::now();

        QueryExecutor executor(storageManager);
        std::shared_ptr<Table> result = executor.execute(query);

        if (result)
        {
            const auto &columns = result->getHeaders();
            const auto &columnData = result->getData(); // Now returns unordered_map<string, vector<string>>
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
                    switch (columnTypes.at(colName))
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
            std::string outputPath = "Team7_query.csv";
            CSVProcessor::saveCSV(outputPath, result->getHeaders(), columnData, columnTypes); // CSVProcessor needs to be updated too
            std::cout << "Saved output to '" << outputPath << "'\n";

            // auto end = high_resolution_clock::now();

            // auto duration = duration_cast<milliseconds>(end - start);

            // std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
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
        std::cerr << "Usage: load <table_name> <filepath>\n";
        return;
    }

    const std::string &tableName = args[0];
    const std::string &filepath = args[1];

    // auto start = high_resolution_clock::now();

    try
    {
        storageManager->loadTable(tableName, filepath);
        // std::cout << "Loaded table '" << tableName << "' from " << filepath << "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error loading table: " << e.what() << "\n";
    }
    // auto end = high_resolution_clock::now();

    // auto duration = duration_cast<milliseconds>(end - start);

    // std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
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

void CommandLineInterface::handleTestCommand()
{
    std::cout << "=== Running Test Cases ===" << std::endl;

    // Step 1: Load tables
    try
    {
        std::cout << "Loading tables..." << std::endl;
        storageManager->loadTable("people", "../../data/input/people.csv");
        std::cout << "Loaded table 'people'" << std::endl;

        storageManager->loadTable("departments", "../../data/input/departments.csv");
        std::cout << "Loaded table 'departments'" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error loading test tables: " << e.what() << "\n";
    }

    // Step 2: Define test queries
    std::vector<std::pair<int, std::string>> testQueries = {
        // ORDER BY
        // {1, "SELECT * FROM people ORDER BY age ASC"},
        // {2, "SELECT * FROM people ORDER BY salary DESC"},
        // {3, "SELECT * FROM people ORDER BY name ASC"},
        // {4, "SELECT * FROM people ORDER BY birthday DESC"},
        // FILTERING
        // {5, "SELECT * FROM people WHERE age > 30"},
        // {6, "SELECT * FROM people WHERE salary >= 60000"},
        // {7, "SELECT * FROM people WHERE name != 'Osama'"},
        // {8, "SELECT * FROM people WHERE birthday < '2000-01-01'"},
        // {9, "SELECT * FROM people WHERE status = 'active'"},
        // NESTED QUERIES
        {10, "SELECT * FROM people WHERE salary > (SELECT AVG(salary) FROM people)"},
        // {11, "SELECT * FROM people WHERE age = (SELECT MAX(age) FROM people)"},
        // JOIN
        // {12, "SELECT p.id, p.name, d.name AS dept_name FROM people p, departments d WHERE p.id % 100 = d.id"},
        // {13, "SELECT p.id, p.name, d.name FROM people p, departments d WHERE p.salary >= d.id * 1000"},
        // MULTIPLE TABLES
        // {14, "SELECT p.name, d.name AS dept, m.name AS manager FROM people p, departments d, people m WHERE p.id % 100 = d.id AND m.id = d.id"},
        // AGGREGATION
        // {15, "SELECT COUNT(*) AS total_people FROM people"},
        // {16, "SELECT AVG(salary) AS avg_salary FROM people"},
        // {17, "SELECT MAX(age) AS max_age FROM people"},
        // {18, "SELECT MIN(birthday) AS earliest_birthday FROM people"},
        // {19, "SELECT name, salary FROM people"}
    };

    // Step 3: Execute each query and verify against test cases
    int totalTests = testQueries.size();
    int passedTests = 0;

    std::cout << "Running " << totalTests << " test queries..." << std::endl;

    for (size_t i = 0; i < testQueries.size(); ++i)
    {
        std::cout << "\nTest Case #" << (i + 1) << ": " << testQueries[i].second << std::endl;

        try
        {
            // Execute the query
            auto start = high_resolution_clock::now();
            QueryExecutor executor(storageManager);
            std::shared_ptr<Table> result = executor.execute(testQueries[i].second);
            auto end = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(end - start);

            if (!result)
            {
                std::cerr << "Query returned no results!" << std::endl;
                continue;
            }

            // Save result to a temporary file
            std::string tempOutputPath = "../../data/output/test_output_temp.csv";
            CSVProcessor::saveCSV(tempOutputPath, result->getHeaders(), result->getData(), result->getColumnTypes());

            // Determine which reference file to compare against
            std::string referenceFilePath;
            referenceFilePath = "../../data/output/test_results/test_case" + std::to_string(testQueries[i].first) + ".csv";

            // Compare with reference file
            bool matched = compareCSVFiles(tempOutputPath, referenceFilePath);

            if (matched)
            {
                std::cout << "✓ TEST PASSED (" << duration.count() << " ms)" << std::endl;
                passedTests++;
            }
            else
            {
                std::cout << "✗ TEST FAILED (" << duration.count() << " ms)" << std::endl;
                std::cout << "  - Results do not match reference file: " << referenceFilePath << std::endl;
            }

            // Clean up temporary files
            // std::remove(tempOutputPath.c_str());

            // Clean up batch tables after each test
            cleanupBatchTables();
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error executing test query: " << e.what() << "\n";
        }
    }

    // Summary
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "Passed: " << passedTests << " / " << totalTests << std::endl;
    double passRate = (static_cast<double>(passedTests) / totalTests) * 100.0;
    std::cout << "Pass Rate: " << passRate << "%" << std::endl;
}

class CSVComparator
{
private:
    // Helper function to trim whitespace from a string
    static std::string trim(const std::string &str)
    {
        auto start = std::find_if_not(str.begin(), str.end(), [](unsigned char c)
                                      { return std::isspace(c); });
        auto end = std::find_if_not(str.rbegin(), str.rend(), [](unsigned char c)
                                    { return std::isspace(c); })
                       .base();
        return (start < end) ? std::string(start, end) : std::string();
    }

    // Helper function to normalize field values
    static std::string normalizeField(const std::string &field)
    {
        std::string normalized = trim(field);

        // Treat empty and zero values as equivalent
        if (normalized.empty() || normalized == "0" || normalized == "0.0" ||
            normalized == "0.00" || normalized == "0.000000")
        {
            return "";
        }

        // Try to parse as numeric value to standardize format
        try
        {
            if (normalized.find('.') != std::string::npos)
            {
                // It's likely a floating point number
                double value = std::stod(normalized);
                // Check if it's an integer value stored as float
                if (value == static_cast<int>(value))
                {
                    return std::to_string(static_cast<int>(value));
                }
                else
                {
                    // Remove trailing zeros for better comparison
                    std::ostringstream oss;
                    oss << value;
                    return oss.str();
                }
            }
            else if (!normalized.empty() &&
                     std::all_of(normalized.begin(), normalized.end(),
                                 [](unsigned char c)
                                 { return std::isdigit(c); }))
            {
                // It's likely an integer
                int value = std::stoi(normalized);
                return std::to_string(value);
            }
        }
        catch (...)
        {
            // Not a number, return as is
        }

        return normalized;
    }

    // Parse a CSV line into vector of fields with handling for trailing commas
    static std::vector<std::string> parseCSVLine(const std::string &line, size_t expectedColumns = 0)
    {
        std::vector<std::string> fields;

        // Handle special case of trailing commas
        std::string processedLine = line;
        if (!processedLine.empty() && processedLine.back() == ',')
        {
            processedLine += " "; // Add space to ensure trailing comma creates an empty field
        }

        std::stringstream ss(processedLine);
        std::string field;

        while (getline(ss, field, ','))
        {
            fields.push_back(normalizeField(field));
        }

        // If we know expected columns, pad with empty strings if needed
        if (expectedColumns > 0 && fields.size() < expectedColumns)
        {
            fields.resize(expectedColumns, "");
        }

        return fields;
    }

public:
    static bool compareCSVFiles(const std::string &file1, const std::string &file2, bool strictMode = false)
    {
        try
        {
            // Load both files
            std::ifstream f1(file1);
            std::ifstream f2(file2);

            if (!f1.is_open())
            {
                std::cerr << "Error: Could not open file " << file1 << std::endl;
                return false;
            }

            if (!f2.is_open())
            {
                std::cerr << "Error: Could not open file " << file2 << std::endl;
                return false;
            }

            std::string line1, line2;
            std::vector<std::vector<std::string>> content1, content2;

            // First pass to determine column count
            size_t maxColumns = 0;
            std::string headerLine;

            // Get header line to count columns
            if (std::getline(f2, headerLine))
            {
                std::vector<std::string> headerFields = parseCSVLine(headerLine);
                maxColumns = headerFields.size();
                content2.push_back(headerFields);
            }

            // Read the header from file1 as well
            if (std::getline(f1, line1))
            {
                content1.push_back(parseCSVLine(line1, maxColumns));
            }

            // Read remaining lines
            while (std::getline(f1, line1))
            {
                if (!line1.empty())
                {
                    content1.push_back(parseCSVLine(line1, maxColumns));
                }
            }

            while (std::getline(f2, line2))
            {
                if (!line2.empty())
                {
                    content2.push_back(parseCSVLine(line2, maxColumns));
                }
            }

            // Basic check: number of rows should match
            if (content1.size() != content2.size())
            {
                std::cout << "  - Different number of rows. Expected: " << content2.size()
                          << ", Got: " << content1.size() << std::endl;
                return false;
            }

            // Before comparing rows, ensure all rows have the same number of columns
            for (auto &row : content1)
            {
                if (row.size() < maxColumns)
                {
                    row.resize(maxColumns, "");
                }
            }

            for (auto &row : content2)
            {
                if (row.size() < maxColumns)
                {
                    row.resize(maxColumns, "");
                }
            }

            // Compare each row
            for (size_t i = 0; i < content1.size(); ++i)
            {
                auto &row1 = content1[i];
                auto &row2 = content2[i];

                // Check if rows have the same number of columns (should be guaranteed now)
                if (row1.size() != row2.size())
                {
                    std::cout << "  - Different number of columns at row " << i << ". "
                              << "Expected: " << row2.size() << ", Got: " << row1.size() << std::endl;

                    // Print out the actual row content for debugging
                    std::cout << "  - Row " << i << " content in file 1: ";
                    for (const auto &field : row1)
                    {
                        std::cout << "'" << field << "', ";
                    }
                    std::cout << std::endl;

                    std::cout << "  - Row " << i << " content in file 2: ";
                    for (const auto &field : row2)
                    {
                        std::cout << "'" << field << "', ";
                    }
                    std::cout << std::endl;

                    return false;
                }

                // Compare each field
                for (size_t j = 0; j < row1.size(); ++j)
                {
                    if (strictMode)
                    {
                        // Direct string comparison in strict mode
                        if (row1[j] != row2[j])
                        {
                            std::cout << "  - Mismatch at row " << i << ", column " << j << ":" << std::endl;
                            std::cout << "    Expected: '" << row2[j] << "'" << std::endl;
                            std::cout << "    Got: '" << row1[j] << "'" << std::endl;
                            return false;
                        }
                    }
                    else
                    {
                        // Special handling for comparing numeric values (salary column)
                        std::string norm1 = normalizeField(row1[j]);
                        std::string norm2 = normalizeField(row2[j]);

                        if (norm1 != norm2)
                        {
                            // Try to parse as numbers for salary column (typically column 3)
                            if (j == 3)
                            {
                                try
                                {
                                    double val1 = norm1.empty() ? 0.0 : std::stod(norm1);
                                    double val2 = norm2.empty() ? 0.0 : std::stod(norm2);

                                    // Consider them equal if they're very close
                                    if (std::abs(val1 - val2) < 0.01)
                                    {
                                        continue;
                                    }
                                }
                                catch (...)
                                {
                                    // Not comparable as numbers, fall back to string comparison
                                }
                            }

                            std::cout << "  - Mismatch at row " << i << ", column " << j << ":" << std::endl;
                            std::cout << "    Expected: '" << row2[j] << "' (normalized: '" << norm2 << "')" << std::endl;
                            std::cout << "    Got: '" << row1[j] << "' (normalized: '" << norm1 << "')" << std::endl;
                            return false;
                        }
                    }
                }
            }

            std::cout << "CSV files are semantically equivalent." << std::endl;
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error comparing files: " << e.what() << "\n";
            return false;
        }
    }
};

bool CommandLineInterface::compareCSVFiles(const std::string &file1, const std::string &file2)
{
    return CSVComparator::compareCSVFiles(file1, file2, false);
}

void CommandLineInterface::displayHelp()
{
    std::cout << "Available commands:\n"
              << "  load <table_name> <filepath> - Load a CSV file as a table\n"
              << "  set mode <GPU>               - List all loaded tables\n"
              << "  <SQL query>                  - Execute a query\n"
              << "  test                         - Run Test Script\n"
              << "  help                         - Show this help\n"
              << "  exit/quit                    - Exit the program\n";
}