#include "DataHandling/CSVProcessor.hpp"
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <filesystem>
#include <sys/stat.h>

void ensureDirectoryExists(const std::string &dirPath) {
    namespace fs = std::filesystem;
    std::error_code ec;
    // create_directories will recursively create all missing parents.
    if (!fs::create_directories(dirPath, ec) && ec) {
        throw std::runtime_error(
          "Failed to create directory '" + dirPath + "': " + ec.message());
    }
}

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

void CSVProcessor::saveCSV(const std::string &filepath, const std::vector<std::string> &headers, const std::vector<std::vector<std::string>> &rows)
{
    ensureDirectoryExists("../../data/output/");
    std::ofstream out(filepath);
    if (!out.is_open())
    {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }

    // Write headers
    for (size_t i = 0; i < headers.size(); ++i)
    {
        out << headers[i];
        if (i < headers.size() - 1)
            out << ",";
    }
    out << "\n";

    // Write rows
    for (const auto &row : rows)
    {
        for (size_t i = 0; i < row.size(); ++i)
        {
            out << row[i];
            if (i < row.size() - 1)
                out << ",";
        }
        out << "\n";
    }

    out.close();
}