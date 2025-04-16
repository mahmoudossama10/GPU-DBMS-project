#include "Utilities/StringUtils.hpp"
#include <algorithm>
#include <sstream>
#include <cstring>

namespace StringUtils
{

    std::string trim(const std::string &str)
    {
        auto start = str.begin();
        while (start != str.end() && std::isspace(*start))
        {
            start++;
        }

        auto end = str.end();
        while (end != start && std::isspace(*(end - 1)))
        {
            end--;
        }

        return std::string(start, end);
    }

    std::string trim_left(const std::string &str)
    {
        auto start = str.begin();
        while (start != str.end() && std::isspace(*start))
        {
            start++;
        }
        return std::string(start, str.end());
    }

    std::string trim_right(const std::string &str)
    {
        auto end = str.end();
        while (end != str.begin() && std::isspace(*(end - 1)))
        {
            end--;
        }
        return std::string(str.begin(), end);
    }

    bool iequals(const std::string &a, const std::string &b)
    {
        return a.size() == b.size() &&
               std::equal(a.begin(), a.end(), b.begin(),
                          [](char c1, char c2)
                          {
                              return std::tolower(c1) == std::tolower(c2);
                          });
    }

    std::vector<std::string> split(const std::string &str,
                                   const std::string &delimiters,
                                   bool keepEmpty)
    {
        std::vector<std::string> tokens;
        std::string::size_type pos, lastPos = 0;

        while ((pos = str.find_first_of(delimiters, lastPos)) != std::string::npos)
        {
            if (pos != lastPos || keepEmpty)
            {
                tokens.emplace_back(str.substr(lastPos, pos - lastPos));
            }
            lastPos = pos + 1;
        }

        if (lastPos < str.size() || keepEmpty)
        {
            tokens.emplace_back(str.substr(lastPos));
        }

        return tokens;
    }

    std::string join(const std::vector<std::string> &parts,
                     const std::string &delimiter)
    {
        std::ostringstream oss;
        for (size_t i = 0; i < parts.size(); ++i)
        {
            if (i != 0)
                oss << delimiter;
            oss << parts[i];
        }
        return oss.str();
    }

    bool startsWith(const std::string &str, const std::string &prefix)
    {
        return str.size() >= prefix.size() &&
               str.compare(0, prefix.size(), prefix) == 0;
    }

    bool endsWith(const std::string &str, const std::string &suffix)
    {
        return str.size() >= suffix.size() &&
               str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
    }
}