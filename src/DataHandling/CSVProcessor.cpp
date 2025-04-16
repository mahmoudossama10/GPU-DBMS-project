#include "DataHandling/CSVProcessor.hpp"
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <filesystem>

CSVProcessor::CSVData CSVProcessor::loadCSV(const std::string &filepath)
{

    std::ifstream file(filepath);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    std::vector<std::string> headers;
    std::vector<std::vector<std::string>> data;
    std::string line;

    // Read headers
    if (std::getline(file, line))
    {
        headers = parseCSVLine(line);
    }

    // Read data rows
    while (std::getline(file, line))
    {
        auto row = parseCSVLine(line);
        if (row.size() != headers.size())
        {
            throw std::runtime_error("CSV row has different column count than header");
        }
        data.push_back(row);
    }

    return {headers, data};
}

void CSVProcessor::writeCSV(const std::string &filepath,
                            const std::vector<std::string> &headers,
                            const std::vector<std::vector<std::string>> &data)
{
    std::ofstream file(filepath);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to create file: " + filepath);
    }

    // Write headers
    file << StringUtils::join(headers, ",") << "\n";

    // Write data
    for (const auto &row : data)
    {
        file << StringUtils::join(row, ",") << "\n";
    }
}

std::vector<std::string> CSVProcessor::parseCSVLine(const std::string &line)
{
    std::vector<std::string> result;
    std::stringstream ss(line);
    std::string field;

    while (std::getline(ss, field, ','))
    {
        result.push_back(StringUtils::trim(field));
    }

    return result;
}