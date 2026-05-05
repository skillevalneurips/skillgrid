---
name: excel
description: Use this skill whenever the user wants to read, write, analyze, or manipulate Excel files (.xlsx, .xls). This includes reading spreadsheets, extracting data from specific sheets or cell ranges, creating new workbooks, modifying existing ones, applying formatting, working with formulas, creating charts, and converting between Excel and other formats like CSV or JSON. If the user mentions a .xlsx or .xls file, use this skill.
license: Proprietary. LICENSE.txt has complete terms
---

# Excel File Processing Guide

## Overview

This guide covers reading, writing, and manipulating Excel files (.xlsx, .xls) using Python libraries. The primary library is `openpyxl` for .xlsx files, with `xlrd` for legacy .xls files and `pandas` for data analysis workflows.

## Quick Start

```python
import openpyxl

# Read an Excel file
wb = openpyxl.load_workbook("data.xlsx")
ws = wb.active
print(f"Sheet: {ws.title}, Rows: {ws.max_row}, Cols: {ws.max_column}")

# Read all data
for row in ws.iter_rows(values_only=True):
    print(row)
```

## Dependencies

```bash
pip install openpyxl pandas xlrd
```

## Python Libraries

### openpyxl - Read and Write .xlsx

#### Read an Excel File
```python
import openpyxl

wb = openpyxl.load_workbook("data.xlsx")

# List all sheet names
print(wb.sheetnames)

# Access a specific sheet
ws = wb["Sheet1"]

# Read a specific cell
value = ws["A1"].value
print(f"A1: {value}")

# Read a range of cells
for row in ws.iter_rows(min_row=1, max_row=10, min_col=1, max_col=5, values_only=True):
    print(row)
```

#### Read All Sheets
```python
wb = openpyxl.load_workbook("data.xlsx")

for sheet_name in wb.sheetnames:
    ws = wb[sheet_name]
    print(f"\n--- {sheet_name} ---")
    for row in ws.iter_rows(values_only=True):
        print(row)
```

#### Write an Excel File
```python
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Results"

# Write headers
headers = ["Name", "Score", "Grade"]
ws.append(headers)

# Write data rows
data = [
    ["Alice", 95, "A"],
    ["Bob", 82, "B"],
    ["Charlie", 78, "C"],
]
for row in data:
    ws.append(row)

wb.save("output.xlsx")
```

#### Modify an Existing File
```python
wb = openpyxl.load_workbook("data.xlsx")
ws = wb.active

# Update a cell
ws["B2"] = 100

# Add a new row
ws.append(["New Entry", 88, "B+"])

wb.save("data_modified.xlsx")
```

### pandas - Data Analysis with Excel

#### Read Excel to DataFrame
```python
import pandas as pd

# Read default sheet
df = pd.read_excel("data.xlsx")
print(df.head())

# Read a specific sheet
df = pd.read_excel("data.xlsx", sheet_name="Sheet2")

# Read all sheets into a dict of DataFrames
all_sheets = pd.read_excel("data.xlsx", sheet_name=None)
for name, df in all_sheets.items():
    print(f"\n--- {name} ---")
    print(df.head())
```

#### Read Specific Rows/Columns
```python
# Skip header rows
df = pd.read_excel("data.xlsx", skiprows=2)

# Read specific columns
df = pd.read_excel("data.xlsx", usecols="A:C")

# Read specific columns by name
df = pd.read_excel("data.xlsx", usecols=["Name", "Score"])

# Set a specific row as header
df = pd.read_excel("data.xlsx", header=1)
```

#### Write DataFrame to Excel
```python
import pandas as pd

df = pd.DataFrame({
    "Name": ["Alice", "Bob"],
    "Score": [95, 82]
})

# Write to Excel
df.to_excel("output.xlsx", index=False)

# Write multiple sheets
with pd.ExcelWriter("output.xlsx") as writer:
    df.to_excel(writer, sheet_name="Scores", index=False)
    df.describe().to_excel(writer, sheet_name="Summary")
```

#### Analyze Excel Data
```python
df = pd.read_excel("data.xlsx")

# Basic statistics
print(df.describe())

# Filter rows
filtered = df[df["Score"] > 80]

# Group and aggregate
grouped = df.groupby("Category")["Value"].sum()

# Pivot table
pivot = pd.pivot_table(df, values="Sales", index="Region", columns="Quarter", aggfunc="sum")
```

### xlrd - Read Legacy .xls Files

```python
import xlrd

wb = xlrd.open_workbook("legacy.xls")
ws = wb.sheet_by_index(0)

print(f"Sheet: {ws.name}, Rows: {ws.nrows}, Cols: {ws.ncols}")

for row_idx in range(ws.nrows):
    row = [ws.cell_value(row_idx, col_idx) for col_idx in range(ws.ncols)]
    print(row)
```

## Common Tasks

### Convert Excel to CSV
```python
import pandas as pd

df = pd.read_excel("data.xlsx")
df.to_csv("data.csv", index=False)
```

### Convert Excel to JSON
```python
import pandas as pd

df = pd.read_excel("data.xlsx")
df.to_json("data.json", orient="records", indent=2)
```

### Merge Multiple Excel Files
```python
import pandas as pd
import glob

files = glob.glob("reports/*.xlsx")
dfs = [pd.read_excel(f) for f in files]
combined = pd.concat(dfs, ignore_index=True)
combined.to_excel("combined.xlsx", index=False)
```

### Extract Data from Specific Cell Ranges
```python
import openpyxl

wb = openpyxl.load_workbook("data.xlsx")
ws = wb.active

# Extract a named range or specific area
data = []
for row in ws.iter_rows(min_row=2, max_row=50, min_col=1, max_col=4, values_only=True):
    if any(cell is not None for cell in row):
        data.append(row)

print(f"Extracted {len(data)} rows")
```

## Quick Reference

| Task | Best Tool | Example |
|------|-----------|---------|
| Read .xlsx | openpyxl | `openpyxl.load_workbook("file.xlsx")` |
| Read .xls | xlrd | `xlrd.open_workbook("file.xls")` |
| Data analysis | pandas | `pd.read_excel("file.xlsx")` |
| Write .xlsx | openpyxl/pandas | `wb.save()` or `df.to_excel()` |
| Convert to CSV | pandas | `df.to_csv("out.csv")` |
| Multiple sheets | pandas | `pd.read_excel(f, sheet_name=None)` |
