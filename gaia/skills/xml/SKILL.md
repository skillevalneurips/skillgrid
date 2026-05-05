---
name: xml
description: Use this skill whenever the user wants to read, parse, query, create, modify, or validate XML files. This includes extracting data from XML documents, navigating XML trees, using XPath queries, transforming XML with XSLT, converting XML to other formats (JSON, CSV, dict), handling namespaces, and working with large XML files via streaming parsers. If the user mentions a .xml file or asks about XML data, use this skill.
license: Proprietary. LICENSE.txt has complete terms
---

# XML File Processing Guide

## Overview

This guide covers reading, parsing, querying, and manipulating XML files using Python's built-in `xml.etree.ElementTree` module (primary) and `lxml` for advanced features like full XPath support, XSLT, and schema validation.

## Quick Start

```python
import xml.etree.ElementTree as ET

tree = ET.parse("data.xml")
root = tree.getroot()

print(f"Root tag: {root.tag}")
for child in root:
    print(f"  {child.tag}: {child.text}")
```

## Dependencies

```bash
pip install lxml  # Optional, for advanced XPath/XSLT/validation
```

## Python Libraries

### xml.etree.ElementTree (Built-in)

#### Parse an XML File
```python
import xml.etree.ElementTree as ET

tree = ET.parse("data.xml")
root = tree.getroot()

# Root element info
print(f"Tag: {root.tag}")
print(f"Attributes: {root.attrib}")
print(f"Children: {len(list(root))}")
```

#### Parse XML from String
```python
import xml.etree.ElementTree as ET

xml_string = """<?xml version="1.0"?>
<catalog>
    <book id="1">
        <title>Python Basics</title>
        <author>John Doe</author>
        <price>29.99</price>
    </book>
</catalog>"""

root = ET.fromstring(xml_string)
```

#### Navigate the Tree
```python
import xml.etree.ElementTree as ET

tree = ET.parse("data.xml")
root = tree.getroot()

# Iterate over direct children
for child in root:
    print(child.tag, child.attrib)

# Access specific child by index
first_child = root[0]

# Get text content
for elem in root.iter("title"):
    print(elem.text)

# Get attribute
for elem in root.iter("book"):
    print(elem.get("id"))
```

#### Find Elements
```python
import xml.etree.ElementTree as ET

tree = ET.parse("data.xml")
root = tree.getroot()

# Find first match
title = root.find(".//title")
print(title.text if title is not None else "Not found")

# Find all matches
all_titles = root.findall(".//title")
for t in all_titles:
    print(t.text)

# Find with attributes
book = root.find(".//book[@id='1']")
```

#### Extract All Data to Dictionary
```python
import xml.etree.ElementTree as ET

def xml_to_dict(element):
    """Recursively convert an XML element to a dictionary."""
    result = {}
    if element.attrib:
        result["@attributes"] = element.attrib

    children = list(element)
    if not children:
        return element.text

    for child in children:
        child_data = xml_to_dict(child)
        if child.tag in result:
            if not isinstance(result[child.tag], list):
                result[child.tag] = [result[child.tag]]
            result[child.tag].append(child_data)
        else:
            result[child.tag] = child_data

    return result

tree = ET.parse("data.xml")
data = xml_to_dict(tree.getroot())
```

#### Create and Write XML
```python
import xml.etree.ElementTree as ET

root = ET.Element("catalog")

book = ET.SubElement(root, "book", id="1")
ET.SubElement(book, "title").text = "Python Basics"
ET.SubElement(book, "author").text = "John Doe"
ET.SubElement(book, "price").text = "29.99"

tree = ET.ElementTree(root)
ET.indent(tree, space="  ")  # Pretty print (Python 3.9+)
tree.write("output.xml", encoding="unicode", xml_declaration=True)
```

#### Modify XML
```python
import xml.etree.ElementTree as ET

tree = ET.parse("data.xml")
root = tree.getroot()

# Update text
for price in root.iter("price"):
    new_price = float(price.text) * 1.1  # 10% increase
    price.text = f"{new_price:.2f}"

# Add new element
new_book = ET.SubElement(root, "book", id="99")
ET.SubElement(new_book, "title").text = "New Book"

# Remove element
for book in root.findall("book"):
    if book.get("id") == "1":
        root.remove(book)

tree.write("modified.xml", encoding="unicode", xml_declaration=True)
```

### Handle Namespaces
```python
import xml.etree.ElementTree as ET

tree = ET.parse("data.xml")
root = tree.getroot()

# Register namespaces to preserve them on output
namespaces = {
    "ns": "http://example.com/namespace"
}

for prefix, uri in namespaces.items():
    ET.register_namespace(prefix, uri)

# Find with namespace
for elem in root.findall(".//ns:item", namespaces):
    print(elem.text)
```

### lxml - Advanced XML Processing

#### XPath Queries
```python
from lxml import etree

tree = etree.parse("data.xml")

# Full XPath support
results = tree.xpath("//book[price > 30]/title/text()")
for title in results:
    print(title)

# XPath with predicates
expensive = tree.xpath("//book[price > 50]")
first_book = tree.xpath("//book[1]")
```

### Streaming Parser for Large XML Files

```python
import xml.etree.ElementTree as ET

# iterparse for memory-efficient processing of large files
for event, elem in ET.iterparse("large.xml", events=("end",)):
    if elem.tag == "record":
        # Process each record
        name = elem.findtext("name")
        value = elem.findtext("value")
        print(f"{name}: {value}")

        elem.clear()  # Free memory
```

## Common Tasks

### Convert XML to JSON
```python
import xml.etree.ElementTree as ET
import json

def elem_to_dict(elem):
    result = {}
    if elem.attrib:
        result.update(elem.attrib)
    for child in elem:
        child_data = elem_to_dict(child) if len(child) else child.text
        if child.tag in result:
            if not isinstance(result[child.tag], list):
                result[child.tag] = [result[child.tag]]
            result[child.tag].append(child_data)
        else:
            result[child.tag] = child_data
    return result

tree = ET.parse("data.xml")
data = elem_to_dict(tree.getroot())

with open("data.json", "w") as f:
    json.dump(data, f, indent=2)
```

### Convert XML to CSV
```python
import xml.etree.ElementTree as ET
import csv

tree = ET.parse("data.xml")
root = tree.getroot()

# Assuming flat record structure
records = root.findall(".//record")
if records:
    fields = [child.tag for child in records[0]]

    with open("data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for record in records:
            row = [record.findtext(field, "") for field in fields]
            writer.writerow(row)
```

### Count Elements
```python
import xml.etree.ElementTree as ET
from collections import Counter

tree = ET.parse("data.xml")
root = tree.getroot()

tag_counts = Counter(elem.tag for elem in root.iter())
for tag, count in tag_counts.most_common():
    print(f"{tag}: {count}")
```

## Quick Reference

| Task | Best Tool | Example |
|------|-----------|---------|
| Parse XML | ElementTree | `ET.parse("file.xml")` |
| Find elements | ElementTree | `root.findall(".//tag")` |
| XPath queries | lxml | `tree.xpath("//book[price>30]")` |
| Create XML | ElementTree | `ET.Element()`, `ET.SubElement()` |
| Large files | iterparse | `ET.iterparse("large.xml")` |
| Namespaces | ElementTree | `findall(".//ns:tag", namespaces)` |
| To JSON | manual | Recursive `elem_to_dict()` |
| To CSV | csv + ET | Extract fields, write rows |
