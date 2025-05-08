#include "./CLI/CommandLineInterface.hpp"
#include <duckdb.hpp>

#include <iostream>

#include <iostream>
#include <regex>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stack>
#include <chrono>

using namespace std;
using namespace std::chrono;

std::string trim(const std::string &str)
{
    const std::string whitespace = " \t\r\n";
    const auto start = str.find_first_not_of(whitespace);
    if (start == std::string::npos)
        return "";
    const auto end = str.find_last_not_of(whitespace);
    return str.substr(start, end - start + 1);
}

std::string readSQLQueryFromFile(const std::string &filepath)
{
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    std::string line;
    std::ostringstream query_stream;

    while (std::getline(file, line))
    {
        query_stream << trim(line) << ' ';
    }

    file.close();
    std::string query = query_stream.str();

    // Final clean-up: collapse multiple spaces into one
    query.erase(std::unique(query.begin(), query.end(),
                            [](char a, char b)
                            { return a == ' ' && b == ' '; }),
                query.end());

    return query;
}

vector<string> extractTableNames(const string &sql);

void processTableReference(const string &ref, vector<string> &tables)
{
    string trimmed = trim(ref);
    if (trimmed.empty())
        return;

    if (trimmed[0] == '(')
    {
        size_t start = trimmed.find('(');
        size_t end = trimmed.rfind(')');
        if (start != string::npos && end != string::npos && end > start)
        {
            string subquery = trimmed.substr(start + 1, end - start - 1);
            vector<string> subTables = extractTableNames(subquery);
            tables.insert(tables.end(), subTables.begin(), subTables.end());
        }
    }
    else
    {
        istringstream iss(trimmed);
        string tableName;
        iss >> tableName;
        tables.push_back(tableName);
    }
}

vector<string> splitTableReferences(const string &clause)
{
    vector<string> refs;
    string current;
    int parenLevel = 0;

    for (char c : clause)
    {
        if (c == '(')
            parenLevel++;
        else if (c == ')')
            parenLevel--;

        if (c == ',' && parenLevel == 0)
        {
            refs.push_back(trim(current));
            current.clear();
        }
        else
        {
            current += c;
        }
    }
    if (!current.empty())
        refs.push_back(trim(current));

    return refs;
}

vector<string> extractTableNames(const string &sql)
{
    vector<string> tables;
    regex from_regex("\\bfrom\\b", regex::icase);
    sregex_iterator it(sql.begin(), sql.end(), from_regex);

    for (; it != sregex_iterator(); ++it)
    {
        size_t from_pos = it->position() + it->length();
        size_t end_pos = sql.find_first_of(";)", from_pos);
        if (end_pos == string::npos)
            end_pos = sql.length();

        string clause = sql.substr(from_pos, end_pos - from_pos);
        vector<string> refs = splitTableReferences(clause);

        for (const string &ref : refs)
        {
            processTableReference(ref, tables);
        }
    }

    // Remove duplicates
    sort(tables.begin(), tables.end());
    tables.erase(unique(tables.begin(), tables.end()), tables.end());

    return tables;
}

int main(int argc, char *argv[])
{

    auto start = high_resolution_clock::now();

    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <inputDirectory> <queryFilePath>" << std::endl;
        return 1;
    }

    try
    {
        CommandLineInterface cli;

        std::string inputDirectory = argv[1];
        std::string queryFilePath = argv[2];

        std::string query = readSQLQueryFromFile(queryFilePath);

        if (!inputDirectory.empty() && inputDirectory.back() != '/')
        {
            inputDirectory += "/";
        }

        auto tableNames = extractTableNames(query);

        for (const auto &table : tableNames)
        {
            std::vector<std::string> loadCommand = {table, inputDirectory + table + ".csv"};
            cli.handleLoadCommand(loadCommand);
        }

        cli.storageManager->inputDirectory = inputDirectory;

        // Extract the file name from queryFilePath and trim whitespace
        std::string outputFileName = queryFilePath;
        // Remove directory path
        size_t lastSlash = outputFileName.find_last_of("/\\");
        if (lastSlash != std::string::npos)
        {
            outputFileName = outputFileName.substr(lastSlash + 1);
        }
        // Remove file extension if present
        size_t lastDot = outputFileName.find_last_of('.');
        if (lastDot != std::string::npos)
        {
            outputFileName = outputFileName.substr(0, lastDot);
        }
        // Trim whitespace
        outputFileName = trim(outputFileName);

        cli.processQuery(query, outputFileName);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    auto end = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end - start);

    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
    return 0;
}

// int main()
// {
//     try
//     {
//         CommandLineInterface cli;

//         std::string inputDirectory = "../../data/input/";
//         cli.storageManager->inputDirectory = inputDirectory;
//         cli.run();
//     }
//     catch (const std::exception &e)
//     {
//         std::cerr << "Fatal error: " << e.what() << std::endl;
//         return 1;
//     }

//     return 0;
// }