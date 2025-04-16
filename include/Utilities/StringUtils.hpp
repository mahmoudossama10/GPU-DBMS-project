#pragma once

#include <string>
#include <vector>
#include <cctype>

namespace StringUtils
{

    // Trim whitespace from both ends of a string
    std::string trim(const std::string &str);
    std::string trim_left(const std::string &str);
    std::string trim_right(const std::string &str);

    // Case-insensitive string comparison
    bool iequals(const std::string &a, const std::string &b);

    // String splitting with multiple delimiters
    std::vector<std::string> split(const std::string &str,
                                   const std::string &delimiters = " ",
                                   bool keepEmpty = false);

    // Join strings with a delimiter
    std::string join(const std::vector<std::string> &parts,
                     const std::string &delimiter = ", ");

    // Check if string starts/ends with specific substring
    bool startsWith(const std::string &str, const std::string &prefix);
    bool endsWith(const std::string &str, const std::string &suffix);

    // Convert string to lower/upper case
    std::string toLower(const std::string &str);
    std::string toUpper(const std::string &str);

    // Replace all occurrences of a substring
    std::string replaceAll(std::string str,
                           const std::string &from,
                           const std::string &to);

} // namespace StringUtils