#pragma once
#include <string>
#include <memory>
#include <cstdlib>

#include "../DataHandling/StorageManager.hpp"
#include "../QueryProcessing/QueryExecutor.hpp"

class CommandLineInterface
{
public:
    CommandLineInterface();
    void run();

private:
    void processQuery(const std::string &query);
    void handleLoadCommand(const std::vector<std::string> &args);
    void handleShowTablesCommand();
    void displayHelp();
    void cleanupBatchTables();

    std::shared_ptr<StorageManager> storageManager;
};