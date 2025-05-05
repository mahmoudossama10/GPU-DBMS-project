import random
import string
from datetime import datetime, timedelta

N = 1000

# Seed for reproducibility
random.seed(42)

# Base list of names (extend with suffixes for uniqueness)
base_names = [
    "Ahmed", "Ali", "Mohamed", "Omar", "Hassan", "Ibrahim", "Khaled", "Mahmoud",
    "Youssef", "Tarek", "Amr", "Kareem", "Sami", "Ziad", "Fadi", "Nour",
    "Lina", "Sara", "Hala", "Reem", "Dina", "Nadia", "Aya", "Yara",
    "AOZ", "MoA", "YUKI", "Abdelrahman", "Fatima", "Zainab", "Mariam"
]

# Generate unique names
names = []
used_names = set()
for i in range(N):
    while True:
        if random.random() < 0.7 and base_names:
            name = random.choice(base_names)
        else:
            name = ''.join(random.choices(string.ascii_uppercase, k=3))
        if name in used_names:
            name = f"{name}{i+1}"
        if name not in used_names:
            used_names.add(name)
            names.append(name)
            break

# Generate data
data = []
start_date = datetime(2020, 1, 1, 0, 0, 0)
end_date = datetime(2025, 5, 4, 23, 59, 59)
time_range = (end_date - start_date).total_seconds()

for i in range(N):
    id_val = i + 1
    name = names[i]
    # Age between 3.0 and 30.0, with 1 to 4 decimal places
    age = random.uniform(3.0, 30.0)
    decimal_places = random.randint(1, 4)
    age_str = f"{age:.{decimal_places}f}"
    # Salary as integer between 30000 and 100000
    salary = random.randint(30000, 100000)
    # Random datetime
    random_seconds = random.uniform(0, time_range)
    random_date = start_date + timedelta(seconds=random_seconds)
    date_str = random_date.strftime("%Y-%m-%d %H:%M:%S")
    data.append(f"{id_val},{name},{age_str},{salary},{date_str}")

# Output header and first few entries
print("id,name,age,salary,birthday")
for row in data[:10]:
    print(row)

# Save full dataset to a file
with open("generated_data.csv", "w") as f:
    f.write("id,name,age,salary,datetime\n")
    for row in data:
        f.write(row + "\n")