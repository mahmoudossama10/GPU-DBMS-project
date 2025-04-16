# GPU DBMS project

for sql-parser
Compile the library make to create libsqlparser.so
(Optional, Recommended) Run make install to copy the library to /usr/local/lib/
Run the tests make test to make sure everything worked
Include the SQLParser.h from src/ (or from /usr/local/lib/hsql/ if you installed it) and link the library in your project

for the main
g++ main.cpp -o main -I/sql-parser-main/src -L/sql-parser-main/build -lsqlparser
make 
make run
load employees ./data/input/employees.csv 

select * from employees