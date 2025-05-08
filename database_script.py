import csv
import random
from datetime import datetime, timedelta

# Seed for reproducibility
random.seed(42)

N = 1000

# Time range for datetime generation
start_date = datetime(2020, 1, 1, 0, 0, 0)
end_date = datetime(2025, 5, 4, 23, 59, 59)
time_range = (end_date - start_date).total_seconds()

def random_datetime_str():
    random_seconds = random.uniform(0, time_range)
    random_date = start_date + timedelta(seconds=random_seconds)
    return random_date.strftime("%Y-%m-%d %H:%M:%S")

# ---- Generate departments.csv ----
departments = ["Engineering", "HR", "Marketing", "Sales", "Finance", 
               "IT", "Operations", "Legal", "R&D", "Support"]

with open('departments.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["department_id", "department_name", "created_at"])
    for i in range(N):
        writer.writerow([
            i+1,
            random.choice(departments),
            random_datetime_str()
        ])

# ---- Generate employees.csv ----
first_names = ["Emma", "Liam", "Olivia", "Noah", "Ava", 
               "William", "Sophia", "James", "Isabella", "Oliver"]
last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", 
              "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]

with open('employees.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["employee_id", "name", "departments_department_id", "salary", "age", "created_at"])
    for i in range(N):
        salary = round(random.uniform(30000.0, 150000.0), random.randint(1, 4))
        writer.writerow([
            i+1,
            f"{random.choice(first_names)} {random.choice(last_names)}",
            random.randint(1, 1000),
            salary,
            random.randint(22, 65),
            random_datetime_str()
        ])

# ---- Generate projects.csv ----
project_names = ["Project Alpha", "Project Beta", "Project Gamma",
                 "Project Delta", "Project Epsilon", "Project Zeta"]

with open('projects.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["project_id", "project_name", "departments_department_id", "created_at"])
    for i in range(N):
        writer.writerow([
            i+1,
            f"{random.choice(project_names)} {random.randint(100, 999)}",
            random.randint(1, 1000),
            random_datetime_str()
        ])
