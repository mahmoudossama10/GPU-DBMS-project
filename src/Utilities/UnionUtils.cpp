#include "Utilities/UnionUtils.hpp"
#include <algorithm>
#include <sstream>
#include <cstring>
#include <string>
#include <sstream>
#include <iomanip>

namespace UnionUtils
{
    std::string valueToString(unionV aggValue, ColumnType aggType)
    {
        switch (aggType)
        {
        case ColumnType::STRING:
            return *(aggValue.s);

        case ColumnType::INTEGER:
            return std::to_string(aggValue.i->value);

        case ColumnType::DOUBLE:
            return std::to_string(aggValue.d->value);

        case ColumnType::DATETIME:
        {
            dateTime *dt = aggValue.t;
            std::ostringstream oss;
            oss << std::setw(4) << std::setfill('0') << dt->year << "-"
                << std::setw(2) << std::setfill('0') << dt->month << "-"
                << std::setw(2) << std::setfill('0') << dt->day << " "
                << std::setw(2) << std::setfill('0') << static_cast<int>(dt->hour) << ":"
                << std::setw(2) << std::setfill('0') << static_cast<int>(dt->minute) << ":"
                << std::setw(2) << std::setfill('0') << static_cast<int>(dt->second);
            return oss.str();
        }

        default:
            return "UNKNOWN";
        }
    }

    unionV copyUnionValue(const unionV &value, ColumnType type)
    {
        unionV copy = {};
        copy.i = new TheInteger();
        copy.d = new TheDouble();

        switch (type)
        {
        case ColumnType::STRING:
            copy.s = (value.s != nullptr) ? new std::string(*value.s) : nullptr;
            break;
        case ColumnType::INTEGER:
            copy.i->value = value.i->value;
            copy.i->is_null = value.i->is_null;
            break;
        case ColumnType::DOUBLE:
            copy.d->value = value.d->value;
            copy.d->is_null = value.d->is_null;
            break;
        case ColumnType::DATETIME:
            if (value.t != nullptr)
            {
                copy.t = new dateTime;
                *copy.t = *value.t;
            }
            else
            {
                copy.t = nullptr;
            }
            break;
        }

        return copy;
    }

}