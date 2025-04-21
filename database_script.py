import csv
import random
from itertools import cycle

# Seed for reproducibility
random.seed(42)

# Generate departments.csv (1000000 rows)
departments = ["Engineering", "HR", "Marketing", "Sales", "Finance", 
               "IT", "Operations", "Legal", "R&D", "Support"]

with open('departments.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["department_id", "department_name"])
    for i in range(1000000):
        writer.writerow([i+1, random.choice(departments)])

# Generate employees.csv (1000000 rows)
first_names = ["Emma", "Liam", "Olivia", "Noah", "Ava", 
              "William", "Sophia", "James", "Isabella", "Oliver"]
last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", 
             "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]

with open('employees.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["employee_id", "name", "department_id", "salary", "age"])
    for i in range(1000000):
        writer.writerow([
            i+1,
            f"{random.choice(first_names)} {random.choice(last_names)}",
            random.randint(1, 1000000),  # department_id
            random.randint(30000, 150000),  # salary
            random.randint(22, 65)  # age
        ])

# Generate projects.csv (1000000 rows)
project_names = ["Project Alpha", "Project Beta", "Project Gamma",
                "Project Delta", "Project Epsilon", "Project Zeta"]

with open('projects.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["project_id", "project_name", "department_id", "start_year"])
    for i in range(1000000):
        writer.writerow([
            i+1,
            f"{random.choice(project_names)} {random.randint(100,999)}",
            random.randint(1, 1000000),  # department_id
            random.randint(2010, 2023)  # start_year
        ])