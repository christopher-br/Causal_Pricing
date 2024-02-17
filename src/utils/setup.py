# LOAD MODULES
import os
from typing import Any, Dict, Tuple, List
import csv
import json
import re
import yaml

# CUSTOM FUNCTIONS
def load_config(file_path: str) -> Dict[str, Any]:
    with open(file_path) as file:
        config = yaml.safe_load(file)
    
    return config

def check_create_csv(
    file_path: str,
    header: Tuple,
) -> None:
    with open(file_path, "a") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        
        # Write header
        if csvfile.tell() == 0:
            writer.writeheader()

def get_rows(
    file_path: str,
) -> List:
    rows = []
    with open(file_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        
        for row in reader:
            parsed_row = []
            for item in row:
                if item.isdigit():
                    parsed_row.append(int(item))
                elif item.replace('.', '', 1).isdigit():
                    parsed_row.append(float(item))
                else:
                    parsed_row.append(item)
            rows.append(tuple(parsed_row))
    
    return rows

def add_row(
    file_path: str,
    row: Tuple,
) -> None:
    with open(file_path, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)

def add_dict(
    file_path: str,
    row: Dict,
) -> None:
    with open(file_path, "a", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        
        # Write header if file is empty
        if csvfile.tell() == 0:
            writer.writeheader()
        
        # Write row
        writer.writerow(row)