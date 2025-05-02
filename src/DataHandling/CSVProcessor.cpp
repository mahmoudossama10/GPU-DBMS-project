#include "DataHandling/CSVProcessor.hpp"
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <filesystem>
#include <regex>
#include <sys/stat.h>

// Default constructor
CSVProcessor::CSVProcessor()
{
    // Default initialization
}
CSVProcessor::CSVData CSVProcessor::loadCSV(const std::string &filepath)
{

    std::ifstream file(filepath);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    std::vector<std::string> headers;
    std::unordered_map<std::string, std::vector<unionV>> columnData;
    std::unordered_map<std::string, ColumnType> columnTypeMap;
    std::string line;

    // Read headers
    if (std::getline(file, line))
    {
        headers = parseCSVLine(line);

        // Initialize column vectors
        for (const auto &header : headers)
        {
            columnData[header] = std::vector<unionV>();
            // Default to STRING type initially
            columnTypeMap[header] = ColumnType::STRING;
        }
    }

    // First pass: Read a sample of data to determine column types
    const size_t SAMPLE_SIZE = 10; // Sample first 10 rows to determine types
    std::vector<std::vector<std::string>> sampleRows;
    size_t sampleCount = 0;

    // Store current position to rewind after sampling
    std::streampos initialPos = file.tellg();

    while (std::getline(file, line) && sampleCount < SAMPLE_SIZE)
    {
        auto row = parseCSVLine(line);
        if (row.size() == headers.size()) // Skip malformed rows
        {
            sampleRows.push_back(row);
            sampleCount++;
        }
    }

    // Analyze sample data to determine column types
    for (size_t i = 0; i < headers.size(); ++i)
    {
        const std::string &header = headers[i];
        bool canBeInt = true;
        bool canBeDouble = true;
        bool canBeBool = true;

        for (const auto &row : sampleRows)
        {
            const std::string &value = row[i];

            // Skip empty values when determining type
            if (value.empty())
                continue;

            // Check if can be boolean
            if (canBeBool)
            {
                std::string valueLower = value;
                std::transform(valueLower.begin(), valueLower.end(), valueLower.begin(),
                               [](unsigned char c)
                               { return std::tolower(c); });
                if (valueLower != "true" && valueLower != "false" &&
                    valueLower != "1" && valueLower != "0" &&
                    valueLower != "yes" && valueLower != "no" &&
                    valueLower != "y" && valueLower != "n")
                {
                    canBeBool = false;
                }
            }

            // Check if can be integer
            if (canBeInt)
            {
                try
                {
                    size_t pos = 0;
                    std::stoi(value, &pos);
                    if (pos != value.length()) // Not the entire string was consumed
                        canBeInt = false;
                }
                catch (...)
                {
                    canBeInt = false;
                }
            }

            // Check if can be double
            if (canBeDouble && !canBeInt) // Try double only if not integer
            {
                try
                {
                    size_t pos = 0;
                    std::stod(value, &pos);
                    if (pos != value.length()) // Not the entire string was consumed
                        canBeDouble = false;
                }
                catch (...)
                {
                    canBeDouble = false;
                }
            }
        }

        // Determine the most specific type that fits all values
        if (canBeInt)
            columnTypeMap[header] = ColumnType::INTEGER;
        else if (canBeDouble)
            columnTypeMap[header] = ColumnType::DOUBLE;
        else
            columnTypeMap[header] = ColumnType::STRING;
    }

    // Rewind file to read full data with determined types
    file.clear(); // Clear any error flags
    file.seekg(initialPos);

    // Read all data rows with the determined types
    size_t rowCount = 0;
    while (std::getline(file, line))
    {
        auto row = parseCSVLine(line);
        if (row.size() != headers.size())
        {
            throw std::runtime_error("CSV row has different column count than header");
        }
        if (rowCount > 106)
        {
            int lol = 5;
            int z = 5;
        }
        // Add each field to its respective column with proper type conversion
        for (size_t i = 0; i < headers.size(); ++i)
        {
            const std::string &header = headers[i];
            ColumnType type = columnTypeMap[header];
            unionV value = convertToUnionV(row[i], type);
            columnData[header].push_back(value);
        }
        rowCount++;
    }

    return {headers, columnData, columnTypeMap};
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

unionV CSVProcessor::convertToUnionV(const std::string &value, ColumnType type)
{
    unionV result;

    switch (type)
    {
    case ColumnType::STRING:
        result.s = new std::string(value);
        break;

    case ColumnType::INTEGER:
        try
        {
            result.i = std::stoll(value);
        }
        catch (const std::exception &)
        {
            result.i = 0; // Default value for failed conversion
        }
        break;

    case ColumnType::DOUBLE:
        try
        {
            result.d = std::stod(value);
        }
        catch (const std::exception &)
        {
            result.d = 0.0; // Default value for failed conversion
        }
        break;

    case ColumnType::DATETIME:
        result.t = parseDateTime(value);
        break;
    }

    return result;
}

std::string CSVProcessor::convertUnionToString(const unionV &value, ColumnType type)
{
    std::string result;

    switch (type)
    {
    case ColumnType::STRING:
        if (value.s != nullptr)
        {
            result = *(value.s);
        }
        break;

    case ColumnType::INTEGER:
        result = std::to_string(value.i);
        break;

    case ColumnType::DOUBLE:
        result = std::to_string(value.d);
        break;

    case ColumnType::DATETIME:
        if (value.t != nullptr)
        {
            dateTime &dt = *(value.t);
            // Format as YYYY-MM-DD HH:MM:SS
            std::ostringstream oss;
            oss << dt.year << "-"
                << std::setw(2) << std::setfill('0') << dt.month << "-"
                << std::setw(2) << std::setfill('0') << dt.day << " "
                << std::setw(2) << std::setfill('0') << static_cast<int>(dt.hour) << ":"
                << std::setw(2) << std::setfill('0') << static_cast<int>(dt.minute) << ":"
                << std::setw(2) << std::setfill('0') << static_cast<int>(dt.second);
            result = oss.str();
        }
        break;
    }

    return result;
}

void CSVProcessor::cleanupUnionV(unionV &value, ColumnType type)
{
    switch (type)
    {
    case ColumnType::STRING:
        delete value.s;
        value.s = nullptr;
        break;

    case ColumnType::DATETIME:
        delete value.t;
        value.t = nullptr;
        break;

    // No cleanup needed for basic types
    case ColumnType::INTEGER:
    case ColumnType::DOUBLE:
        break;
    }
}

dateTime *CSVProcessor::parseDateTime(const std::string &datetime)
{
    dateTime *dt = new dateTime();

    // Initialize with defaults
    dt->year = 1970;
    dt->month = 1;
    dt->day = 1;
    dt->hour = 0;
    dt->minute = 0;
    dt->second = 0;

    // Use regex to parse various datetime formats
    // Expected format: YYYY-MM-DD HH:MM:SS or YYYY/MM/DD HH:MM:SS
    std::regex datetime_pattern(
        R"((\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:\s+(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?)?)",
        std::regex::extended);

    std::smatch matches;
    if (std::regex_match(datetime, matches, datetime_pattern))
    {
        if (matches.size() > 1)
            dt->year = static_cast<unsigned short>(std::stoi(matches[1].str()));
        if (matches.size() > 2)
            dt->month = static_cast<unsigned short>(std::stoi(matches[2].str()));
        if (matches.size() > 3)
            dt->day = static_cast<unsigned short>(std::stoi(matches[3].str()));
        if (matches.size() > 4 && matches[4].matched)
            dt->hour = static_cast<unsigned char>(std::stoi(matches[4].str()));
        if (matches.size() > 5 && matches[5].matched)
            dt->minute = static_cast<unsigned char>(std::stoi(matches[5].str()));
        if (matches.size() > 6 && matches[6].matched)
            dt->second = static_cast<unsigned char>(std::stoi(matches[6].str()));
    }

    return dt;
}

void CSVProcessor::saveCSV(const std::string &filepath,
                           const std::vector<std::string> &headers,
                           const std::unordered_map<std::string, std::vector<unionV>> &columnData,
                           const std::unordered_map<std::string, ColumnType> &columnTypes)
{
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

    // Determine number of rows from the first column
    size_t rowCount = 0;
    if (!headers.empty() && columnData.find(headers[0]) != columnData.end())
    {
        rowCount = columnData.at(headers[0]).size();
    }

    // Write data row by row
    for (size_t rowIdx = 0; rowIdx < rowCount; ++rowIdx)
    {
        for (size_t colIdx = 0; colIdx < headers.size(); ++colIdx)
        {
            const std::string &header = headers[colIdx];

            // Check if column exists and has enough rows
            if (columnData.find(header) != columnData.end() &&
                rowIdx < columnData.at(header).size())
            {
                // Get column type and convert union value to string
                ColumnType type = columnTypes.at(header);
                std::string value = convertUnionToString(columnData.at(header)[rowIdx], type);
                out << value;
            }

            // Add comma if not last column
            if (colIdx < headers.size() - 1)
                out << ",";
        }
        out << "\n";
    }

    out.close();
}