#pragma once
#include <string>
#include <vector>
#include <fstream>
#include "Utilities/StringUtils.hpp"

class CSVProcessor
{
public:
    using CSVData = std::pair<std::vector<std::string>, std::vector<std::vector<std::string>>>;

    static CSVData loadCSV(const std::string &filepath);
    static void writeCSV(const std::string &filepath,
                         const std::vector<std::string> &headers,
                         const std::vector<std::vector<std::string>> &data);

private:
    static std::vector<std::string> parseCSVLine(const std::string &line);
};