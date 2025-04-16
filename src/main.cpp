#include "./CLI/CommandLineInterface.hpp"
#include <iostream>

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
    return 0;
}