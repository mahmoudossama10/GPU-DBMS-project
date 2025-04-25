import csv

def escape_sql_string(val):
    return val.replace("'", "''")  # Escape single quotes by doubling them

def csv_to_sql(csv_file_path, table_name, output_sql_path):
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)

        with open(output_sql_path, 'w', encoding='utf-8') as sqlfile:
            for row in reader:
                values = []
                for val in row:
                    val = val.strip()
                    if val.lower() in ('null', ''):
                        values.append('NULL')
                    else:
                        escaped_val = escape_sql_string(val)
                        values.append(f"'{escaped_val}'")
                values_str = ", ".join(values)
                sql = f"INSERT INTO {table_name} ({', '.join(headers)}) VALUES ({values_str});\n"
                sqlfile.write(sql)

    print(f"SQL statements written to {output_sql_path}")

csv_to_sql('projects_new.csv', 'projects', 'projects.sql')
