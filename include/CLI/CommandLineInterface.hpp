#pragma once
#include <string>
#include <memory>
#include <cstdlib>

#include "../DataHandling/StorageManager.hpp"
#include "../QueryProcessing/QueryExecutor.hpp"

#include "../../libs/linenoise-ng-master/include/linenoise.h"

class CommandLineInterface
{
public:
    CommandLineInterface();
    ~CommandLineInterface();
    void run();

private:
    void processQuery(const std::string &query);
    void handleLoadCommand(const std::vector<std::string> &args);
    void handleShowTablesCommand();
    void displayHelp();
    std::string getInput();
    void cleanupBatchTables();
    void handleTestCommand();
    bool compareCSVFiles(const std::string &file1, const std::string &file2);

    std::shared_ptr<StorageManager> storageManager;
    std::vector<std::string> commandHistory;
    size_t historyIndex;
};