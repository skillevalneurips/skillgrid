---
name: txt
description: Use this skill whenever the user wants to read, write, parse, search, or manipulate plain text files (.txt). This includes reading file contents, searching for patterns with regex, counting words/lines/characters, extracting structured data from unstructured text, handling encodings, processing log files, and performing text transformations. If the user mentions a .txt file or needs to process plain text data, use this skill.
license: Proprietary. LICENSE.txt has complete terms
---

# Text File Processing Guide

## Overview

This guide covers reading, writing, searching, and manipulating plain text files using Python's built-in capabilities and the `re` module for pattern matching.

## Quick Start

```python
# Read entire file
with open("data.txt", "r") as f:
    content = f.read()
print(f"Length: {len(content)} chars, {len(content.splitlines())} lines")
```

## Reading Text Files

### Read Entire File
```python
with open("data.txt", "r") as f:
    content = f.read()
```

### Read Lines
```python
# As a list
with open("data.txt", "r") as f:
    lines = f.readlines()  # Includes newline characters

# Stripped lines
with open("data.txt", "r") as f:
    lines = [line.strip() for line in f]

# Iterate line by line (memory efficient)
with open("data.txt", "r") as f:
    for line in f:
        process(line.strip())
```

### Read with Encoding
```python
# UTF-8 (default on most systems)
with open("data.txt", "r", encoding="utf-8") as f:
    content = f.read()

# Latin-1 / ISO-8859-1
with open("data.txt", "r", encoding="latin-1") as f:
    content = f.read()

# Auto-detect encoding
import chardet

with open("data.txt", "rb") as f:
    raw = f.read()
    detected = chardet.detect(raw)
    content = raw.decode(detected["encoding"])
```

### Read Specific Lines
```python
# Read first N lines
with open("data.txt", "r") as f:
    first_10 = [next(f).strip() for _ in range(10)]

# Read lines by range
with open("data.txt", "r") as f:
    lines = f.readlines()
    subset = lines[10:20]  # Lines 11-20 (0-indexed)

# Read last N lines (for small files)
with open("data.txt", "r") as f:
    lines = f.readlines()
    last_10 = lines[-10:]
```

## Writing Text Files

### Write Content
```python
with open("output.txt", "w") as f:
    f.write("Hello World\n")
    f.write("Second line\n")

# Write multiple lines
lines = ["Line 1", "Line 2", "Line 3"]
with open("output.txt", "w") as f:
    f.write("\n".join(lines))

# Append to file
with open("output.txt", "a") as f:
    f.write("Appended line\n")
```

## Searching and Pattern Matching

### Simple Search
```python
with open("data.txt", "r") as f:
    for i, line in enumerate(f, 1):
        if "search_term" in line:
            print(f"Line {i}: {line.strip()}")
```

### Regex Search
```python
import re

with open("data.txt", "r") as f:
    content = f.read()

# Find all matches
emails = re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", content)
urls = re.findall(r"https?://\S+", content)
numbers = re.findall(r"\b\d+\.?\d*\b", content)

# Find with context
for match in re.finditer(r"error.*", content, re.IGNORECASE):
    print(f"Position {match.start()}: {match.group()}")
```

### Extract Structured Data
```python
import re

with open("data.txt", "r") as f:
    content = f.read()

# Key-value pairs (e.g., "Name: John")
pairs = re.findall(r"^(\w+):\s*(.+)$", content, re.MULTILINE)
data = dict(pairs)

# Tabular data (whitespace-separated)
with open("data.txt", "r") as f:
    lines = f.readlines()
    header = lines[0].split()
    rows = [line.split() for line in lines[1:] if line.strip()]
```

## Text Analysis

### Word/Line/Character Counts
```python
with open("data.txt", "r") as f:
    content = f.read()

lines = content.splitlines()
words = content.split()
chars = len(content)

print(f"Lines: {len(lines)}")
print(f"Words: {len(words)}")
print(f"Characters: {chars}")
```

### Word Frequency
```python
from collections import Counter
import re

with open("data.txt", "r") as f:
    content = f.read().lower()

words = re.findall(r"\b\w+\b", content)
freq = Counter(words)

for word, count in freq.most_common(20):
    print(f"{word}: {count}")
```

## Common Tasks

### Process Log Files
```python
import re

entries = []
with open("app.log", "r") as f:
    for line in f:
        match = re.match(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (\w+): (.+)", line)
        if match:
            entries.append({
                "timestamp": match.group(1),
                "level": match.group(2),
                "message": match.group(3)
            })

# Filter errors
errors = [e for e in entries if e["level"] == "ERROR"]
```

### Compare Two Text Files
```python
with open("file1.txt") as f1, open("file2.txt") as f2:
    lines1 = f1.readlines()
    lines2 = f2.readlines()

import difflib
diff = difflib.unified_diff(lines1, lines2, fromfile="file1.txt", tofile="file2.txt")
print("".join(diff))
```

### Replace Text in File
```python
with open("data.txt", "r") as f:
    content = f.read()

content = content.replace("old_text", "new_text")

# Or with regex
import re
content = re.sub(r"pattern", "replacement", content)

with open("data.txt", "w") as f:
    f.write(content)
```

### Split File into Sections
```python
with open("data.txt", "r") as f:
    content = f.read()

# Split by delimiter
sections = content.split("---")

# Split by blank lines
import re
paragraphs = re.split(r"\n\s*\n", content)
```

## Quick Reference

| Task | Method | Example |
|------|--------|---------|
| Read file | `open().read()` | `f.read()` |
| Read lines | `open().readlines()` | `f.readlines()` |
| Write file | `open("w").write()` | `f.write(text)` |
| Search | `in` operator | `"term" in line` |
| Regex search | `re.findall()` | `re.findall(pattern, text)` |
| Word count | `str.split()` | `len(content.split())` |
| Encoding | `encoding` param | `open(f, encoding="utf-8")` |
| Diff | `difflib` | `difflib.unified_diff()` |
