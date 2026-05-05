---
name: json
description: Use this skill whenever the user wants to read, write, parse, query, validate, or transform JSON files. This includes loading JSON data, navigating nested structures, extracting specific fields, converting JSON to other formats (CSV, XML, DataFrame), handling JSON Lines (JSONL), merging JSON files, and working with JSON APIs. If the user mentions a .json or .jsonl file, use this skill.
license: Proprietary. LICENSE.txt has complete terms
---

# JSON File Processing Guide

## Overview

This guide covers reading, writing, querying, and manipulating JSON files using Python's built-in `json` module and complementary tools for data analysis and transformation.

## Quick Start

```python
import json

# Read a JSON file
with open("data.json", "r") as f:
    data = json.load(f)

print(type(data))
print(json.dumps(data, indent=2)[:500])
```

## Reading JSON Files

### Load from File
```python
import json

with open("data.json", "r") as f:
    data = json.load(f)
```

### Load from String
```python
import json

json_string = '{"name": "Alice", "score": 95}'
data = json.loads(json_string)
```

### Read JSON Lines (JSONL)
```python
import json

records = []
with open("data.jsonl", "r") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

print(f"Loaded {len(records)} records")
```

### Handle Encoding Issues
```python
import json

with open("data.json", "r", encoding="utf-8-sig") as f:
    data = json.load(f)
```

## Writing JSON Files

### Write to File
```python
import json

data = {"name": "Alice", "scores": [95, 88, 92]}

with open("output.json", "w") as f:
    json.dump(data, f, indent=2)
```

### Write JSON Lines
```python
import json

records = [
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"},
]

with open("output.jsonl", "w") as f:
    for record in records:
        f.write(json.dumps(record) + "\n")
```

### Custom Serialization
```python
import json
from datetime import datetime

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

data = {"timestamp": datetime.now(), "value": 42}

with open("output.json", "w") as f:
    json.dump(data, f, cls=CustomEncoder, indent=2)
```

## Querying and Navigating JSON

### Access Nested Data
```python
import json

with open("data.json", "r") as f:
    data = json.load(f)

# Dot-style access with safe navigation
def get_nested(data, *keys, default=None):
    """Safely access nested dictionary keys."""
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, default)
        elif isinstance(data, list) and isinstance(key, int):
            data = data[key] if key < len(data) else default
        else:
            return default
    return data

value = get_nested(data, "users", 0, "address", "city")
```

### Filter and Extract
```python
import json

with open("data.json", "r") as f:
    data = json.load(f)

# Filter list of objects
if isinstance(data, list):
    filtered = [item for item in data if item.get("score", 0) > 80]

# Extract specific fields
names = [item["name"] for item in data if "name" in item]

# Flatten nested structure
def flatten(obj, prefix=""):
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{prefix}.{k}" if prefix else k
            items.update(flatten(v, new_key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            items.update(flatten(v, f"{prefix}[{i}]"))
    else:
        items[prefix] = obj
    return items

flat = flatten(data)
```

### Search for Keys/Values
```python
import json

def find_key(data, target_key):
    """Recursively find all values for a given key."""
    results = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key == target_key:
                results.append(value)
            results.extend(find_key(value, target_key))
    elif isinstance(data, list):
        for item in data:
            results.extend(find_key(item, target_key))
    return results

with open("data.json", "r") as f:
    data = json.load(f)

all_names = find_key(data, "name")
```

## Using pandas with JSON

### JSON to DataFrame
```python
import pandas as pd

# Simple JSON array
df = pd.read_json("data.json")

# Nested JSON
import json
with open("data.json", "r") as f:
    data = json.load(f)

df = pd.json_normalize(data)  # Flattens nested dicts

# JSON Lines
df = pd.read_json("data.jsonl", lines=True)
```

### DataFrame to JSON
```python
import pandas as pd

df = pd.DataFrame({"name": ["Alice", "Bob"], "score": [95, 82]})

# As array of records
df.to_json("output.json", orient="records", indent=2)

# As JSON Lines
df.to_json("output.jsonl", orient="records", lines=True)
```

## Common Tasks

### Merge Multiple JSON Files
```python
import json
import glob

all_data = []
for filepath in glob.glob("data/*.json"):
    with open(filepath, "r") as f:
        data = json.load(f)
        if isinstance(data, list):
            all_data.extend(data)
        else:
            all_data.append(data)

with open("merged.json", "w") as f:
    json.dump(all_data, f, indent=2)
```

### Convert JSON to CSV
```python
import json
import csv

with open("data.json", "r") as f:
    data = json.load(f)

# Assuming list of flat dicts
if isinstance(data, list) and data:
    keys = data[0].keys()
    with open("data.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)
```

### Pretty Print / Reformat
```python
import json

with open("data.json", "r") as f:
    data = json.load(f)

# Pretty print to console
print(json.dumps(data, indent=2, ensure_ascii=False))

# Reformat file
with open("data.json", "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
```

### Validate JSON Structure
```python
import json

def validate_json(filepath):
    """Check if a file contains valid JSON."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        return True, data
    except json.JSONDecodeError as e:
        return False, str(e)

valid, result = validate_json("data.json")
if not valid:
    print(f"Invalid JSON: {result}")
```

### Compare Two JSON Files
```python
import json

with open("file1.json") as f1, open("file2.json") as f2:
    data1 = json.load(f1)
    data2 = json.load(f2)

if data1 == data2:
    print("Files are identical")
else:
    # Find differences (for dicts)
    if isinstance(data1, dict) and isinstance(data2, dict):
        keys1, keys2 = set(data1.keys()), set(data2.keys())
        print(f"Only in file1: {keys1 - keys2}")
        print(f"Only in file2: {keys2 - keys1}")
        for key in keys1 & keys2:
            if data1[key] != data2[key]:
                print(f"Different value for '{key}'")
```

## Quick Reference

| Task | Method | Example |
|------|--------|---------|
| Read JSON | `json.load()` | `json.load(open("f.json"))` |
| Read JSONL | line-by-line | `json.loads(line)` per line |
| Write JSON | `json.dump()` | `json.dump(data, f, indent=2)` |
| Parse string | `json.loads()` | `json.loads(string)` |
| Pretty print | `json.dumps()` | `json.dumps(data, indent=2)` |
| To DataFrame | pandas | `pd.read_json()` or `json_normalize()` |
| To CSV | csv + json | `DictWriter` from loaded data |
| Nested access | helper func | `get_nested(data, "a", "b")` |
| Merge files | json + glob | Load all, extend list, dump |
