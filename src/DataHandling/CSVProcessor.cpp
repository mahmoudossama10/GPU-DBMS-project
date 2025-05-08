#include "DataHandling/CSVProcessor.hpp"
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <filesystem>
#include <regex>
#include <sys/stat.h>
#include <omp.h>

// Default constructor
CSVProcessor::CSVProcessor()
{
    // Default initialization
}

// Ensure you have a matching inferTypes function adapted from your friend's approach

std::vector<ColumnType> CSVProcessor::inferTypes(const std::string &row)
{

    std::vector<ColumnType> types;

    auto fields = parseCSVLine(row);

    for (const auto &field : fields)
    {

        try
        {

            auto int_val = std::stoll(field);

            types.push_back(ColumnType::INTEGER);

            continue;
        }
        catch (...)
        {
        }

        try
        {

            auto double_val = std::stod(field);

            types.push_back(ColumnType::DOUBLE);

            continue;
        }
        catch (...)
        {
        }

        types.push_back(ColumnType::STRING);
    }

    return types;
}

CSVProcessor::CSVData CSVProcessor::loadCSV(const std::string &filepath)

{

    // Step 1: Read all lines into memory

    std::vector<std::string> lines;

    std::ifstream file(filepath);

    if (!file.is_open())

    {

        throw std::runtime_error("Unable to open file: " + filepath);
    }

    std::string line;

    while (std::getline(file, line))

    {

        if (!line.empty() && line.back() == '\r')

        {

            line.pop_back();
        }

        if (!line.empty())

        {

            lines.push_back(line);
        }
    }

    file.close();

    if (lines.empty())

    {

        return {std::vector<std::string>(), std::unordered_map<std::string, std::vector<unionV>>(), std::unordered_map<std::string, ColumnType>()};
    }

    // Step 2: Parse headers from the first line and clean them (remove annotations)

    std::vector<std::string> rawHeaders = parseCSVLine(lines[0]);

    std::vector<std::string> headers;

    for (size_t i = 0; i < rawHeaders.size(); ++i)

    {

        std::string cleanedHeader = rawHeaders[i];

        // Remove annotations like (N), (P), (T), (D) from the header name

        size_t pos = cleanedHeader.find('(');

        if (pos != std::string::npos)

        {

            cleanedHeader = cleanedHeader.substr(0, pos);
        }

        // Trim any trailing whitespace

        while (!cleanedHeader.empty() && std::isspace(cleanedHeader.back()))

        {

            cleanedHeader.pop_back();
        }

        headers.push_back(cleanedHeader);
    }

    // Step 3: Overwrite the CSV file with cleaned headers to ensure subsequent loads see updated names

    // Write to a temporary file first to avoid data loss

    std::string tempFilepath = filepath + ".tmp";

    std::ofstream tempFile(tempFilepath);

    if (!tempFile.is_open())

    {

        throw std::runtime_error("Unable to create temporary file for writing: " + tempFilepath);
    }

    // Write the cleaned headers as the first line

    for (size_t i = 0; i < headers.size(); ++i)

    {

        tempFile << headers[i];

        if (i < headers.size() - 1)

            tempFile << ",";
    }

    tempFile << "\n";

    // Write the rest of the lines (data rows) unchanged

    for (size_t i = 1; i < lines.size(); ++i)

    {

        tempFile << lines[i];

        if (i < lines.size() - 1)

            tempFile << "\n";
    }

    tempFile.close();

    // Replace the original file with the temporary file

    if (std::remove(filepath.c_str()) != 0)

    {

        throw std::runtime_error("Failed to remove original file: " + filepath);
    }

    if (std::rename(tempFilepath.c_str(), filepath.c_str()) != 0)

    {

        throw std::runtime_error("Failed to rename temporary file to: " + filepath);
    }

    // Step 4: Infer types from the second row (or a small sample for robustness)

    std::vector<ColumnType> types(headers.size(), ColumnType::STRING);

    if (lines.size() > 1)

    {

        // Use a small sample (up to 5 rows) for type inference, balancing speed and accuracy

        const size_t SAMPLE_SIZE = 5;

        std::vector<std::vector<std::string>> sampleRows;

        for (size_t i = 1; i < lines.size() && sampleRows.size() < SAMPLE_SIZE; ++i)

        {

            auto row = parseCSVLine(lines[i]);

            if (row.size() == headers.size())

            {

                sampleRows.push_back(row);
            }
        }

        for (size_t col = 0; col < headers.size(); ++col)

        {

            bool canBeInt = true;

            bool canBeDouble = true;

            for (const auto &row : sampleRows)

            {

                const std::string &value = row[col];

                if (value.empty())

                    continue;

                if (canBeInt)

                {

                    try

                    {

                        size_t pos = 0;

                        std::stoi(value, &pos);

                        if (pos != value.length())

                            canBeInt = false;
                    }

                    catch (...)

                    {

                        canBeInt = false;
                    }
                }

                if (canBeDouble && !canBeInt)

                {

                    try

                    {

                        size_t pos = 0;

                        std::stod(value, &pos);

                        if (pos != value.length())

                            canBeDouble = false;
                    }

                    catch (...)

                    {

                        canBeDouble = false;
                    }
                }
            }

            if (canBeInt)

                types[col] = ColumnType::INTEGER;

            else if (canBeDouble)

                types[col] = ColumnType::DOUBLE;

            else

                types[col] = ColumnType::STRING;
        }
    }

    // Step 5: Pre-allocate storage for data (array-like for parallelism)

    size_t rowCount = lines.size() > 1 ? lines.size() - 1 : 0;

    std::vector<std::vector<unionV>> tempData(headers.size());

    for (size_t col = 0; col < headers.size(); ++col)

    {

        tempData[col].resize(rowCount); // Pre-allocate space for each column
    }

    // Step 6: Parallel processing of rows into pre-allocated vectors

#pragma omp parallel for schedule(static)

    for (size_t i = 1; i < lines.size(); ++i)

    {

        auto vals = parseCSVLine(lines[i]);

        if (vals.size() != headers.size())

        {

#pragma omp critical

            {

                throw std::runtime_error("Wrong number of columns in file: " + filepath + ", line " + std::to_string(i + 1));
            }
        }

        for (size_t col = 0; col < headers.size(); ++col)

        {

            unionV value = {};

            value.i = new TheInteger();

            value.d = new TheDouble();

            switch (types[col])

            {

            case ColumnType::INTEGER:

                try

                {

                    value.i->value = std::stoll(vals[col]);
                }

                catch (...)

                {

                    value.i->value = 0; // Default value

                    value.i->is_null = true;
                }

                break;

            case ColumnType::DOUBLE:

                try

                {

                    value.d->value = std::stod(vals[col]);
                }

                catch (...)

                {

                    value.d->value = 0.0; // Default value

                    value.d->is_null = true;
                }

                break;

            case ColumnType::STRING:

                value.s = new std::string(vals[col]);

                break;

            case ColumnType::DATETIME:

                value.t = parseDateTime(vals[col]);

                break;
            }

            tempData[col][i - 1] = value; // Direct indexing, thread-safe due to pre-allocation
        }
    }

    // Step 7: Convert temporary array structure to CSVData format

    std::unordered_map<std::string, std::vector<unionV>> columnData;

    std::unordered_map<std::string, ColumnType> columnTypeMap;

    for (size_t col = 0; col < headers.size(); ++col)

    {

        columnData[headers[col]] = std::move(tempData[col]); // Move pre-allocated vector

        columnTypeMap[headers[col]] = types[col];
    }

    return {headers, columnData, columnTypeMap};
}

std::vector<std::string> CSVProcessor::parseCSVLine(const std::string &line)
{
    std::vector<std::string> result;

    std::string field;

    bool inQuotes = false;

    for (size_t i = 0; i < line.length(); ++i)

    {

        char c = line[i];

        if (inQuotes)

        {

            if (c == '"')

            {

                if (i + 1 < line.length() && line[i + 1] == '"')

                {

                    field += '"'; // Escaped quote

                    ++i;
                }

                else

                {

                    inQuotes = false; // End of quoted field
                }
            }

            else

            {

                field += c;
            }
        }

        else

        {

            if (c == '"')

            {

                inQuotes = true;
            }

            else if (c == ',')

            {

                result.push_back(field);

                field.clear();
            }

            else

            {

                field += c;
            }
        }
    }

    result.push_back(field); // Add the last field

    return result;
}

unionV CSVProcessor::convertToUnionV(const std::string &value, ColumnType type)
{
    unionV result = {};
    result.i = new TheInteger();
    result.d = new TheDouble();

    switch (type)
    {
    case ColumnType::STRING:
        result.s = new std::string(value);
        break;

    case ColumnType::INTEGER:
        try
        {
            result.i->value = std::stoll(value);
        }
        catch (const std::exception &)
        {
            result.i->value = 0; // Default value for failed conversion
            result.i->is_null = true;
        }
        break;

    case ColumnType::DOUBLE:
        try
        {
            result.d->value = std::stod(value);
        }
        catch (const std::exception &)
        {
            result.d->value = 0.0; // Default value for failed conversion
            result.d->is_null = true;
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
        if (value.i->is_null == true)
        {
            result = "";
        }
        else
        {
            result = std::to_string(value.i->value);
        }
        break;

    case ColumnType::DOUBLE:
        if (value.d->is_null == true)
        {
            result = "";
        }
        else
        {
            result = std::to_string(value.d->value);
        }
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