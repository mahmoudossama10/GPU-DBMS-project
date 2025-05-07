#ifndef CSV_PROCESSOR_HPP
#define CSV_PROCESSOR_HPP

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include "Table.hpp"

// Forward declaration
namespace StringUtils
{
    std::string join(const std::vector<std::string> &elements, const std::string &delimiter);
    std::string trim(const std::string &str);
}

class CSVProcessor
{
public:
    // Structure for row-major CSV data
    struct CSVData
    {
        std::vector<std::string> headers;
        std::unordered_map<std::string, std::vector<unionV>> data;
        std::unordered_map<std::string, ColumnType> columnTypes;
    };

    // Constructors
    CSVProcessor();                         // Default constructor
    CSVProcessor(size_t rows, size_t cols); // Constructor with table size

    // Methods for row-major format
    static CSVData loadCSV(const std::string &filepath);
    static std::vector<ColumnType> inferTypes(const std::string& row);

    static void saveCSV(const std::string &filepath,
                        const std::vector<std::string> &headers,
                        const std::unordered_map<std::string, std::vector<unionV>> &columnData,
                        const std::unordered_map<std::string, ColumnType> &columnTypes);

    // Helper methods
    static std::vector<std::string> parseCSVLine(const std::string &line);

    // Helper method to convert string to appropriate union type
    static unionV convertToUnionV(const std::string &value, ColumnType type);

    // Helper method to convert union to string for saving
    static std::string convertUnionToString(const unionV &value, ColumnType type);

    // Clean up union memory (for string and datetime pointers)
    static void cleanupUnionV(unionV &value, ColumnType type);

private:
    // Helper for parsing datetime strings
    static dateTime *parseDateTime(const std::string &datetime);
};

#endif // CSV_PROCESSOR_HPP