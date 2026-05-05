---
name: docx
description: Use this skill whenever the user wants to read, write, create, or manipulate Word documents (.docx). This includes extracting text and tables from Word files, creating new documents with formatted text, headings, tables, and images, modifying existing documents, converting DOCX to other formats (text, PDF, HTML), and working with document styles and formatting. If the user mentions a .docx file, use this skill.
license: Proprietary. LICENSE.txt has complete terms
---

# DOCX (Word Document) Processing Guide

## Overview

This guide covers reading, writing, and manipulating Microsoft Word (.docx) files using `python-docx`. DOCX files are ZIP archives containing XML, and `python-docx` provides a clean API for working with them.

## Quick Start

```python
from docx import Document

doc = Document("report.docx")

# Read all text
for paragraph in doc.paragraphs:
    print(paragraph.text)
```

## Dependencies

```bash
pip install python-docx pandas
```

## Reading DOCX Files

### Extract All Text
```python
from docx import Document

doc = Document("report.docx")

full_text = []
for paragraph in doc.paragraphs:
    if paragraph.text.strip():
        full_text.append(paragraph.text)

text = "\n".join(full_text)
print(text)
```

### Extract Text with Style Info
```python
from docx import Document

doc = Document("report.docx")

for para in doc.paragraphs:
    if para.text.strip():
        style = para.style.name
        print(f"[{style}] {para.text}")
```

### Extract Headings Only
```python
from docx import Document

doc = Document("report.docx")

headings = []
for para in doc.paragraphs:
    if para.style.name.startswith("Heading"):
        level = para.style.name.replace("Heading ", "")
        headings.append((level, para.text))
        print(f"H{level}: {para.text}")
```

### Extract Tables
```python
from docx import Document

doc = Document("report.docx")

for i, table in enumerate(doc.tables):
    print(f"\n--- Table {i + 1} ---")
    for row in table.rows:
        cells = [cell.text for cell in row.cells]
        print("\t".join(cells))
```

### Tables to DataFrame
```python
from docx import Document
import pandas as pd

doc = Document("report.docx")

def table_to_df(table):
    """Convert a docx table to a pandas DataFrame."""
    data = []
    for row in table.rows:
        data.append([cell.text for cell in row.cells])
    if data:
        return pd.DataFrame(data[1:], columns=data[0])
    return pd.DataFrame()

for i, table in enumerate(doc.tables):
    df = table_to_df(table)
    print(f"\nTable {i + 1}:")
    print(df)
```

### Extract Images
```python
from docx import Document
import os

doc = Document("report.docx")

os.makedirs("images", exist_ok=True)
for i, rel in enumerate(doc.part.rels.values()):
    if "image" in rel.reltype:
        image = rel.target_part
        ext = os.path.splitext(image.partname)[1]
        with open(f"images/image_{i}{ext}", "wb") as f:
            f.write(image.blob)
        print(f"Saved: images/image_{i}{ext}")
```

### Extract Hyperlinks
```python
from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT

doc = Document("report.docx")

for para in doc.paragraphs:
    for run in para.runs:
        # Check for hyperlinks in the paragraph's XML
        pass

# Alternative: parse XML directly
import xml.etree.ElementTree as ET

for rel in doc.part.rels.values():
    if rel.reltype == RT.HYPERLINK:
        print(f"URL: {rel.target_ref}")
```

## Writing DOCX Files

### Create a New Document
```python
from docx import Document
from docx.shared import Inches, Pt

doc = Document()

# Add heading
doc.add_heading("Report Title", level=0)

# Add paragraph
doc.add_paragraph("This is the introduction paragraph.")

# Add formatted text
para = doc.add_paragraph()
run = para.add_run("Bold text")
run.bold = True
para.add_run(" and ")
run = para.add_run("italic text")
run.italic = True

# Add bullet list
doc.add_paragraph("First item", style="List Bullet")
doc.add_paragraph("Second item", style="List Bullet")
doc.add_paragraph("Third item", style="List Bullet")

# Add numbered list
doc.add_paragraph("Step one", style="List Number")
doc.add_paragraph("Step two", style="List Number")

doc.save("new_report.docx")
```

### Add a Table
```python
from docx import Document

doc = Document()
doc.add_heading("Data Table", level=1)

# Create table
headers = ["Name", "Score", "Grade"]
data = [
    ["Alice", "95", "A"],
    ["Bob", "82", "B"],
    ["Charlie", "78", "C"],
]

table = doc.add_table(rows=1, cols=len(headers))
table.style = "Table Grid"

# Add headers
for i, header in enumerate(headers):
    table.rows[0].cells[i].text = header

# Add data rows
for row_data in data:
    row = table.add_row()
    for i, value in enumerate(row_data):
        row.cells[i].text = value

doc.save("table_report.docx")
```

### Add an Image
```python
from docx import Document
from docx.shared import Inches

doc = Document()
doc.add_heading("Image Report", level=1)
doc.add_picture("chart.png", width=Inches(5))
doc.add_paragraph("Figure 1: Chart showing results.")

doc.save("image_report.docx")
```

## Modifying Existing Documents

### Replace Text
```python
from docx import Document

doc = Document("template.docx")

for para in doc.paragraphs:
    if "{{placeholder}}" in para.text:
        for run in para.runs:
            if "{{placeholder}}" in run.text:
                run.text = run.text.replace("{{placeholder}}", "actual value")

doc.save("filled_template.docx")
```

### Add Content to Existing Document
```python
from docx import Document

doc = Document("existing.docx")

# Append new content
doc.add_heading("New Section", level=1)
doc.add_paragraph("Additional content added programmatically.")

doc.save("updated.docx")
```

## Common Tasks

### Convert DOCX to Plain Text
```python
from docx import Document

doc = Document("report.docx")

text_parts = []
for para in doc.paragraphs:
    text_parts.append(para.text)

# Include table text
for table in doc.tables:
    for row in table.rows:
        row_text = "\t".join(cell.text for cell in row.cells)
        text_parts.append(row_text)

with open("report.txt", "w") as f:
    f.write("\n".join(text_parts))
```

### Get Document Metadata
```python
from docx import Document

doc = Document("report.docx")
props = doc.core_properties

print(f"Title: {props.title}")
print(f"Author: {props.author}")
print(f"Created: {props.created}")
print(f"Modified: {props.modified}")
print(f"Last modified by: {props.last_modified_by}")
```

### Count Words and Pages
```python
from docx import Document

doc = Document("report.docx")

word_count = 0
para_count = 0
for para in doc.paragraphs:
    if para.text.strip():
        para_count += 1
        word_count += len(para.text.split())

print(f"Paragraphs: {para_count}")
print(f"Words: {word_count}")
print(f"Tables: {len(doc.tables)}")
```

## Quick Reference

| Task | Method | Example |
|------|--------|---------|
| Read text | `doc.paragraphs` | `para.text` for each paragraph |
| Read tables | `doc.tables` | `cell.text` for each cell |
| Write doc | `Document()` | `doc.add_paragraph()`, `doc.save()` |
| Add heading | `add_heading()` | `doc.add_heading("Title", level=1)` |
| Add table | `add_table()` | `doc.add_table(rows, cols)` |
| Add image | `add_picture()` | `doc.add_picture("img.png")` |
| Extract images | `part.rels` | Iterate image relationships |
| Metadata | `core_properties` | `doc.core_properties.author` |
| To text | loop paragraphs | Join `para.text` with newlines |
