---
name: info_extraction
description: Use this skill whenever the agent needs to extract, accumulate, or evaluate information from web page content. This includes identifying relevant facts from noisy page content, extracting specific data points (dates, names, addresses, numbers), accumulating information across multiple pages into a memory, judging whether accumulated information is sufficient to answer a query, filtering useful vs. irrelevant content, and generating final answers from collected information. If the task involves analyzing page content for relevant information, use this skill. Answers must be evidence-grounded and should never invent missing fields.
license: Proprietary. LICENSE.txt has complete terms
---

# Information Extraction Guide

## Overview

This guide covers extracting and managing information from web pages in the WebWalker benchmark. The benchmark uses a two-stage extraction pipeline: (1) extract useful information from each page observation, and (2) judge whether accumulated information is sufficient to answer the query. This guide covers both LLM-based and heuristic extraction methods.

## Critical Fixes For Reward Stability

To avoid low-reward runs caused by brittle retrieval and guessed answers:

- Extract only facts explicitly present in collected observations.
- Keep structured memory with required fields instead of free-form notes.
- If a required field is missing, return `unknown` for that field rather than guessing.
- Require source evidence snippets for each extracted fact.
- Reject outputs that contain placeholders (`None`, empty dicts, "not found") as final answers.

## Quick Start

```python
from openai import OpenAI
import json

client = OpenAI(api_key="your-key", base_url="your-server")

def extract_info(query, observation, model="gpt-4o"):
    """Extract evidence-grounded fields from a page observation."""
    schema = {
        "funding_program": "unknown",
        "start_date": "unknown",
        "duration": "unknown",
        "evidence": []
    }
    messages = [
        {"role": "system", "content": (
            "Extract only facts explicitly present in the observation. "
            "Return strict JSON with keys funding_program, start_date, duration, evidence. "
            "If missing, set value to 'unknown'. Never infer from prior knowledge."
        )},
        {"role": "user", "content": f"- Query: {query}\n- Observation: {observation}"}
    ]
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=messages
    )
    payload = response.choices[0].message.content or "{}"
    data = json.loads(payload)
    return {
        "funding_program": data.get("funding_program", schema["funding_program"]),
        "start_date": data.get("start_date", schema["start_date"]),
        "duration": data.get("duration", schema["duration"]),
        "evidence": data.get("evidence", []),
    }
```

## Dependencies

```bash
pip install openai requests
```

## Two-Stage Extraction Pipeline

### Stage 1: Observation Information Extraction

Extract useful information from each page observation relative to the query.

```python
import json
from openai import OpenAI

EXTRACTION_PROMPT = """You are an information extraction agent. Analyze the observation and extract only directly supported facts. Never guess.

**Input:**
- Query: "<Query>"
- Observation: "<Current Observation>"

**Output (JSON):**
{
  "usefulness": true,
  "funding_program": "<value or unknown>",
  "start_date": "<value or unknown>",
  "duration": "<value or unknown>",
  "evidence": ["<short supporting quote 1>", "<quote 2>"]
}
Or, if the observation does not contain useful information:
{
  "usefulness": false
}
Only respond with valid JSON.
"""

def extract_from_observation(client, model, query, observation):
    """Stage 1: Extract useful info from a page observation."""
    user_prompt = f"- Query: {query}\n- Observation: {observation}"
    messages = [
        {"role": "system", "content": EXTRACTION_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=messages
    )
    content = response.choices[0].message.content
    if "true" in content:
        try:
            data = json.loads(content)
            return {
                "funding_program": data.get("funding_program", "unknown"),
                "start_date": data.get("start_date", "unknown"),
                "duration": data.get("duration", "unknown"),
                "evidence": data.get("evidence", []),
            }
        except (json.JSONDecodeError, KeyError):
            return None
    return None
```

### Stage 2: Answer Sufficiency Evaluation

Judge whether accumulated information is enough to answer the query.

```python
ANSWER_EVAL_PROMPT = """You are a query answering agent. Your task is to evaluate whether the accumulated useful information is sufficient to answer the current query. If it is sufficient, return a JSON object with a "judge" value of true and an "answer" field with the answer. If the information is insufficient, return a JSON object with a "judge" value of false.

**Input:**
- Query: "<Query>"
- Accumulated Information: "<Accumulated Useful Information>"

**Output (JSON):**
{
    "judge": true,
    "answer": "<Generated Answer> using string format"
}
Or, if the information is insufficient to answer the query:
{
    "judge": false
}
Only respond with valid JSON.
"""

def evaluate_sufficiency(client, model, query, memory):
    """Stage 2: Check if accumulated info is enough to answer."""
    required = ["funding_program", "start_date", "duration"]
    if not all(memory.get(k) and memory.get(k) != "unknown" for k in required):
        return None
    accumulated = json.dumps(memory, ensure_ascii=False)
    user_prompt = f"- Query: {query}\n- Accumulated Information: {accumulated}"
    messages = [
        {"role": "system", "content": ANSWER_EVAL_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=messages
    )
    content = response.choices[0].message.content
    if "true" in content:
        try:
            return json.loads(content)["answer"]
        except (json.JSONDecodeError, KeyError):
            return None
    return None
```

### Combined Pipeline with Memory
```python
class InformationAccumulator:
    def __init__(self, client, model, query):
        self.client = client
        self.model = model
        self.query = query
        self.memory = {
            "funding_program": "unknown",
            "start_date": "unknown",
            "duration": "unknown",
            "evidence": [],
        }

    def process_observation(self, observation):
        """Process a page observation and update memory."""
        info = extract_from_observation(
            self.client, self.model, self.query, observation
        )
        if info:
            for key in ["funding_program", "start_date", "duration"]:
                value = info.get(key)
                if value and value != "unknown":
                    self.memory[key] = value
            for ev in info.get("evidence", []):
                if ev and ev not in self.memory["evidence"]:
                    self.memory["evidence"].append(ev)
            return True
        return False

    def check_answer(self):
        """Check if we have enough info to answer."""
        return evaluate_sufficiency(
            self.client, self.model, self.query, self.memory
        )

    def get_memory_summary(self):
        """Get formatted memory for display."""
        return json.dumps(self.memory, ensure_ascii=False, indent=2)
```

## Heuristic Extraction Methods

### Extract Dates
```python
import re

def extract_dates(text):
    """Extract date patterns from text."""
    patterns = [
        r'\b\d{4}-\d{2}-\d{2}\b',                    # 2025-03-21
        r'\b\d{1,2}/\d{1,2}/\d{4}\b',                # 03/21/2025
        r'\b(?:January|February|March|April|May|June|July|August|September|'
        r'October|November|December)\s+\d{1,2},?\s+\d{4}\b',  # March 21, 2025
        r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|'
        r'September|October|November|December)\s+\d{4}\b',     # 21 March 2025
    ]
    dates = []
    for pattern in patterns:
        dates.extend(re.findall(pattern, text, re.IGNORECASE))
    return dates
```

### Extract Deadlines
```python
import re

def extract_deadlines(text):
    """Extract deadline mentions with associated dates."""
    lines = text.split("\n")
    deadlines = []
    for i, line in enumerate(lines):
        if any(kw in line.lower() for kw in ["deadline", "due", "submission", "due date"]):
            dates = extract_dates(line)
            if dates:
                deadlines.append({"context": line.strip(), "dates": dates})
            # Also check next line
            elif i + 1 < len(lines):
                dates = extract_dates(lines[i + 1])
                if dates:
                    deadlines.append({
                        "context": f"{line.strip()} {lines[i+1].strip()}",
                        "dates": dates
                    })
    return deadlines
```

### Extract Named Entities
```python
import re

def extract_names(text):
    """Extract person names (simple heuristic: capitalized word pairs)."""
    pattern = r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'
    return list(set(re.findall(pattern, text)))

def extract_emails(text):
    """Extract email addresses."""
    return re.findall(r'[\w.+-]+@[\w-]+\.[\w.]+', text)

def extract_addresses(text):
    """Extract potential addresses (lines with numbers and common address terms)."""
    lines = text.split("\n")
    address_keywords = ["street", "st.", "avenue", "ave.", "road", "rd.",
                       "boulevard", "blvd.", "platz", "plaza", "drive", "dr."]
    addresses = []
    for line in lines:
        line_lower = line.lower()
        if any(kw in line_lower for kw in address_keywords):
            addresses.append(line.strip())
    return addresses
```

### Extract Numbers and Statistics
```python
import re

def extract_numbers_with_context(text, context_window=50):
    """Extract numbers with surrounding context."""
    results = []
    for match in re.finditer(r'\b\d+[,.]?\d*\b', text):
        start = max(0, match.start() - context_window)
        end = min(len(text), match.end() + context_window)
        results.append({
            "number": match.group(),
            "context": text[start:end].strip()
        })
    return results
```

## Keyword-Based Extraction

### Extract Sentences by Keywords
```python
def extract_relevant_sentences(text, keywords):
    """Extract sentences containing any of the given keywords."""
    sentences = re.split(r'[.!?\n]', text)
    relevant = []
    keywords_lower = [kw.lower() for kw in keywords]
    for sentence in sentences:
        if any(kw in sentence.lower() for kw in keywords_lower):
            cleaned = sentence.strip()
            if cleaned:
                relevant.append(cleaned)
    return relevant
```

### Extract Sections by Headers
```python
import re

def extract_section(markdown_text, section_name):
    """Extract content under a specific markdown header."""
    pattern = rf'(?:^|\n)#+\s*{re.escape(section_name)}\s*\n(.*?)(?=\n#+\s|\Z)'
    match = re.search(pattern, markdown_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None
```

### Extract Key-Value Pairs
```python
import re

def extract_key_value_pairs(text):
    """Extract key: value pairs from text."""
    pairs = {}
    for line in text.split("\n"):
        match = re.match(r'^([^:]+):\s*(.+)$', line.strip())
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()
            pairs[key] = value
    return pairs
```

## Error Handling and Retries

### Robust LLM Extraction with Retry
```python
import time
import json

def extract_with_retry(client, model, query, observation, max_retries=10):
    """Extract info with exponential backoff on failure."""
    user_prompt = f"- Query: {query}\n- Observation: {observation}"
    messages = [
        {"role": "system", "content": EXTRACTION_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=messages
            )
            content = response.choices[0].message.content
            if "true" in content:
                try:
                    return json.loads(content)["information"]
                except (json.JSONDecodeError, KeyError):
                    return content
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1 * (2 ** attempt))
            else:
                raise e
```

## Common Tasks

### Process Page in WebWalker Pipeline
```python
def process_page_observation(client, model, query, observation, memory):
    """Full pipeline for processing a single page observation."""
    # Stage 1: Extract relevant info
    info = extract_from_observation(client, model, query, observation)

    if info:
        memory.append(info + "\n")

        # Stage 2: Check if we can answer
        if len(memory) >= 1:
            answer = evaluate_sufficiency(client, model, query, memory)
            if answer:
                return {"status": "answered", "answer": answer, "memory": memory}

        return {"status": "continue", "answer": None, "memory": memory}

    return {"status": "no_info", "answer": None, "memory": memory}
```

### Combine Multi-Source Information
```python
def combine_multi_source(memory_items):
    """Combine information from multiple sources into a coherent answer."""
    combined = []
    for i, item in enumerate(memory_items, 1):
        combined.append(f"Source {i}: {item}")
    return " | ".join(combined)
```

### Truncate Long Content for LLM
```python
def truncate_for_llm(text, max_chars=10000):
    """Truncate long page content to fit LLM context."""
    if len(text) <= max_chars:
        return text
    # Keep beginning and end
    half = max_chars // 2
    return text[:half] + "\n...[truncated]...\n" + text[-half:]
```

## Quick Reference

| Task | Method | Example |
|------|--------|---------|
| Extract from page | LLM Stage 1 | `extract_from_observation(client, model, query, obs)` |
| Check sufficiency | LLM Stage 2 | `evaluate_sufficiency(client, model, query, memory)` |
| Extract dates | Regex | `re.findall(date_pattern, text)` |
| Extract by keywords | String search | `any(kw in sentence for kw in keywords)` |
| Extract section | Regex on headers | `re.search(section_pattern, markdown)` |
| Extract key-value | Regex | `re.match(r"key:\s*value", line)` |
| Manage memory | InformationAccumulator | Append info, check answer |
| Handle errors | Retry with backoff | `time.sleep(1 * (2 ** attempt))` |
