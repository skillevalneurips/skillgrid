---
name: csv
description: Use this skill whenever the user wants to read, write, parse, analyze, or manipulate CSV (Comma-Separated Values) files. This includes reading CSV data, filtering and transforming rows, handling different delimiters (TSV, pipe-separated, etc.), dealing with encoding issues, performing aggregations, and converting CSV to other formats. If the user mentions a .csv or .tsv file, use this skill.
license: Proprietary. LICENSE.txt has complete terms
---

# CSV File Processing Guide

## Overview

This guide covers reading, writing, and manipulating CSV files using Python's built-in `csv` module and `pandas` for data analysis. CSV is the most common plain-text tabular format.

## Quick Start

```python
import csv

# Read a CSV file
with open("data.csv", "r", newline="") as f:
    reader = csv.reader(f)
    headers = next(reader)
    for row in reader:
        print(row)
```

## Dependencies

```bash
pip install pandas  # Optional, for advanced analysis
```

## Python Libraries

### csv (Built-in) - Basic Read/Write

#### Read CSV
```python
import csv

with open("data.csv", "r", newline="") as f:
    reader = csv.reader(f)
    headers = next(reader)
    print(f"Columns: {headers}")
    for row in reader:
        print(row)
```

#### Read CSV as Dictionaries
```python
import csv

with open("data.csv", "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row["Name"], row["Score"])
```

#### Write CSV
```python
import csv

headers = ["Name", "Score", "Grade"]
rows = [
    ["Alice", 95, "A"],
    ["Bob", 82, "B"],
    ["Charlie", 78, "C"],
]

with open("output.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(rows)
```

#### Write CSV from Dictionaries
```python
import csv

data = [
    {"Name": "Alice", "Score": 95},
    {"Name": "Bob", "Score": 82},
]

with open("output.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Name", "Score"])
    writer.writeheader()
    writer.writerows(data)
```

#### Handle Different Delimiters
```python
import csv

# Tab-separated (TSV)
with open("data.tsv", "r", newline="") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        print(row)

# Pipe-separated
with open("data.txt", "r", newline="") as f:
    reader = csv.reader(f, delimiter="|")
    for row in reader:
        print(row)

# Auto-detect delimiter
with open("data.csv", "r") as f:
    sample = f.read(4096)
    dialect = csv.Sniffer().sniff(sample)
    f.seek(0)
    reader = csv.reader(f, dialect)
    for row in reader:
        print(row)
```

#### Handle Encoding Issues
```python
import csv

# Read with specific encoding
with open("data.csv", "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)

# Common encodings: utf-8, utf-8-sig (BOM), latin-1, cp1252
```

### pandas - Data Analysis with CSV

#### Read CSV to DataFrame
```python
import pandas as pd

df = pd.read_csv("data.csv")
print(df.head())
print(df.info())
print(df.describe())
```

#### Read with Options
```python
import pandas as pd

# Custom delimiter
df = pd.read_csv("data.tsv", sep="\t")

# Skip rows, set header
df = pd.read_csv("data.csv", skiprows=2, header=0)

# Specify column types
df = pd.read_csv("data.csv", dtype={"ID": str, "Value": float})

# Handle missing values
df = pd.read_csv("data.csv", na_values=["N/A", "null", ""])

# Read specific columns
df = pd.read_csv("data.csv", usecols=["Name", "Score"])

# Parse dates
df = pd.read_csv("data.csv", parse_dates=["date_column"])
```

#### Filter and Transform
```python
import pandas as pd

df = pd.read_csv("data.csv")

# Filter rows
filtered = df[df["Score"] > 80]

# Add computed column
df["Pass"] = df["Score"] >= 70

# Sort
df_sorted = df.sort_values("Score", ascending=False)

# Group and aggregate
summary = df.groupby("Category").agg(
    count=("Score", "count"),
    mean=("Score", "mean"),
    total=("Value", "sum")
).reset_index()
```

#### Write CSV
```python
import pandas as pd

df.to_csv("output.csv", index=False)

# Custom separator
df.to_csv("output.tsv", sep="\t", index=False)
```

## Common Tasks

### Count Rows and Columns
```python
import csv

with open("data.csv", "r", newline="") as f:
    reader = csv.reader(f)
    headers = next(reader)
    row_count = sum(1 for _ in reader)
    print(f"Columns: {len(headers)}, Rows: {row_count}")
```

### Merge Multiple CSV Files
```python
import pandas as pd
import glob

files = glob.glob("data/*.csv")
dfs = [pd.read_csv(f) for f in files]
combined = pd.concat(dfs, ignore_index=True)
combined.to_csv("combined.csv", index=False)
```

### Convert CSV to JSON
```python
import csv
import json

with open("data.csv", "r", newline="") as f:
    reader = csv.DictReader(f)
    data = list(reader)

with open("data.json", "w") as f:
    json.dump(data, f, indent=2)
```

### Large CSV Files (Chunked Reading)
```python
import pandas as pd

# Process in chunks to avoid memory issues
chunks = pd.read_csv("large_file.csv", chunksize=10000)
results = []
for chunk in chunks:
    processed = chunk[chunk["Value"] > 0]
    results.append(processed)

df = pd.concat(results, ignore_index=True)
```

## Quick Reference

| Task | Best Tool | Example |
|------|-----------|---------|
| Read CSV | csv module | `csv.reader(f)` |
| Read as dicts | csv module | `csv.DictReader(f)` |
| Write CSV | csv module | `csv.writer(f).writerows(data)` |
| Data analysis | pandas | `pd.read_csv("file.csv")` |
| TSV files | csv/pandas | `delimiter="\t"` or `sep="\t"` |
| Large files | pandas | `pd.read_csv(f, chunksize=N)` |
| Merge CSVs | pandas | `pd.concat(dfs)` |
