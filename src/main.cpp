#include "./CLI/CommandLineInterface.hpp"
#include <duckdb.hpp>

#include <iostream>

#include <iostream>
#include <regex>
#include <vector>
#include <string>

std::vector<std::string> extractTableNames(const std::string &query)
{
    std::vector<std::string> tables;
    std::string lowered = query;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), ::tolower);

    size_t pos = 0;
    while ((pos = lowered.find("from", pos)) != std::string::npos)
    {
        // Skip "from" inside quotes or subqueries
        size_t start = pos + 4;
        int paren_count = 0;
        bool in_quotes = false;
        size_t end = start;

        while (end < query.size())
        {
            char c = query[end];
            if (c == '\'')
                in_quotes = !in_quotes;
            else if (!in_quotes)
            {
                if (c == '(')
                    ++paren_count;
                else if (c == ')')
                    --paren_count;
                else if ((paren_count == 0) && query.substr(end, 5) == "where")
                    break;
            }
            ++end;
        }

        std::string from_clause = query.substr(start, end - start);

        // Extract table names (ignore aliases and AS)
        std::regex table_regex(R"(([a-zA-Z_][a-zA-Z0-9_]*)(?:\s+(?:AS\s+)?[a-zA-Z_][a-zA-Z0-9_]*)?)");
        auto begin = std::sregex_iterator(from_clause.begin(), from_clause.end(), table_regex);
        auto finish = std::sregex_iterator();

        for (auto it = begin; it != finish; ++it)
        {
            tables.push_back(it->str(1)); // table name
        }

        pos = end;
    }

    return tables;
}

// int main()
// {
//     std::string query = "SELECT * FROM aoz a, (SELECT * FROM nested1 n1, nested2 AS n2 WHERE n1.id = n2.id) AS subq, aoz2 b WHERE a.name = b.name;";
//     auto tables = extractTableNames(query);
//     for (const auto &t : tables)
//         std::cout << t << std::endl;
// }

int main()
{
    try
    {
        CommandLineInterface cli;
        cli.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    // try
    // {
    //     std::string input_directory = "../../data/input/";
    //     std::string query = "SELECT * FROM aoz a, aoz2 b WHERE a.name = b.name;";

    //     auto tableNames = extractTableNames(query);

    //     CommandLineInterface cli;
    //     cli.run();
    // }
    // catch (const std::exception &e)
    // {
    //     std::cerr << "Fatal error: " << e.what() << std::endl;
    //     return 1;
    // }
    return 0;
}