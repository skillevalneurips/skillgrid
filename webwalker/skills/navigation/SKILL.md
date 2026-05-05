---
name: navigation
description: Use this skill whenever the agent needs to plan or execute multi-step web traversal strategies. This includes deciding which links to follow on a page, implementing breadth-first or depth-first exploration, backtracking when a page lacks relevant information, tracking visited pages to avoid loops, managing navigation state and history, following golden paths through a website, and coordinating multi-source information gathering across different page branches. If the task involves strategically navigating through a website to find information, use this skill.
license: Proprietary. LICENSE.txt has complete terms
---

# Web Navigation Strategy Guide

## Overview

This guide covers strategies for navigating websites in the WebWalker benchmark. Tasks require the agent to traverse from a root URL through multiple pages to find specific information. Navigation involves selecting which links to follow, tracking visited pages, backtracking from dead ends, and coordinating multi-source information gathering.

## Quick Start

```python
from collections import deque

class WebNavigator:
    def __init__(self, root_url, max_steps=10):
        self.root_url = root_url
        self.visited = set()
        self.history = []
        self.max_steps = max_steps
        self.step_count = 0

    def visit(self, url):
        self.visited.add(url)
        self.history.append(url)
        self.step_count += 1

    def should_stop(self):
        return self.step_count >= self.max_steps

    def is_visited(self, url):
        return url in self.visited
```

## Navigation Patterns

### ReAct Navigation Loop

The WebWalker benchmark uses a ReAct (Reason + Act) format for navigation. Each step follows:
1. **Thought**: Reason about what information is needed and which link to follow
2. **Action**: Choose a tool action (e.g., visit a page/click a button)
3. **Action Input**: Provide the button/link to click
4. **Observation**: Process the page content returned

```python
# ReAct prompt format
REACT_TEMPLATE = """
Question: {query}
Thought: I need to find {what_to_find}. Looking at the available buttons,
         "{button_name}" seems most relevant because {reason}.
Action: visit_page
Action Input: {{"button": "{button_name}"}}
Observation: {page_content}
"""
```

### Link Selection Strategy

When multiple links are available, select the most relevant one:

```python
def score_link_relevance(link_text, query_keywords):
    """Score a link's relevance to the query based on keyword overlap."""
    link_words = set(link_text.lower().split())
    query_words = set(kw.lower() for kw in query_keywords)
    overlap = link_words & query_words
    return len(overlap) / max(len(query_words), 1)

def select_best_link(links, query_keywords):
    """Select the most relevant link from available options."""
    scored = [(link, score_link_relevance(link["text"], query_keywords))
              for link in links]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0] if scored else None

# Example
links = [
    {"text": "Call for Papers", "url": "/calls/papers"},
    {"text": "Venue", "url": "/venue"},
    {"text": "Industry Track", "url": "/calls/industry"},
    {"text": "Contact", "url": "/contact"}
]
best = select_best_link(links, ["industry", "track", "deadline"])
# -> {"text": "Industry Track", "url": "/calls/industry"}
```

### Breadth-First Exploration
```python
from collections import deque

def bfs_navigate(root_url, get_links_fn, is_target_fn, max_depth=3):
    """Breadth-first search through website pages."""
    queue = deque([(root_url, 0)])
    visited = set()
    results = []

    while queue:
        url, depth = queue.popleft()
        if url in visited or depth > max_depth:
            continue

        visited.add(url)
        content = get_links_fn(url)  # Returns page content and links

        if is_target_fn(content):
            results.append({"url": url, "depth": depth, "content": content})

        for link in content.get("links", []):
            if link["url"] not in visited:
                queue.append((link["url"], depth + 1))

    return results
```

### Depth-First with Backtracking
```python
def dfs_navigate(url, get_links_fn, is_target_fn, visited=None, max_depth=5, depth=0):
    """Depth-first search with backtracking."""
    if visited is None:
        visited = set()

    if depth > max_depth or url in visited:
        return None

    visited.add(url)
    content = get_links_fn(url)

    if is_target_fn(content):
        return {"url": url, "content": content}

    for link in content.get("links", []):
        result = dfs_navigate(link["url"], get_links_fn, is_target_fn,
                              visited, max_depth, depth + 1)
        if result:
            return result

    return None  # Backtrack
```

## State Management

### Track Navigation History
```python
class NavigationState:
    def __init__(self, root_url):
        self.root_url = root_url
        self.visited = set()
        self.history = []       # Ordered list of visited URLs
        self.memory = []        # Accumulated useful information
        self.button_map = {}    # button_text -> URL mapping

    def visit(self, url, button_text=None):
        self.visited.add(url)
        self.history.append({
            "url": url,
            "button": button_text,
            "step": len(self.history)
        })

    def add_memory(self, info):
        self.memory.append(info)

    def get_unvisited_links(self, available_links):
        return [link for link in available_links
                if link["url"] not in self.visited]

    def can_backtrack(self):
        return len(self.history) > 1

    def backtrack(self):
        if self.can_backtrack():
            self.history.pop()
            return self.history[-1]["url"]
        return self.root_url
```

### Avoid Navigation Loops
```python
def detect_loop(history, window=3):
    """Detect if the agent is visiting the same pages in a loop."""
    if len(history) < window * 2:
        return False
    recent = [h["url"] for h in history[-window:]]
    previous = [h["url"] for h in history[-window*2:-window]]
    return recent == previous

def get_unexplored_links(links, visited):
    """Get links that haven't been visited yet."""
    return [link for link in links if link["url"] not in visited]
```

## Multi-Source Navigation

### Handle Multi-Hop Queries

WebWalker tasks may require information from multiple branches of a website (multi-source). For example: "What is the submission deadline AND the venue address?" requires visiting both the call-for-papers page and the venue page.

```python
class MultiSourceNavigator:
    def __init__(self, root_url, sub_queries):
        self.root_url = root_url
        self.sub_queries = sub_queries  # List of sub-questions
        self.answered = {}              # sub_query -> answer
        self.visited = set()

    def decompose_query(self, query):
        """Break a multi-part question into sub-queries."""
        # Use LLM or heuristic to decompose
        # Example: "deadline AND venue" -> ["deadline", "venue"]
        parts = query.split(" and ")
        return [p.strip() for p in parts]

    def is_complete(self):
        """Check if all sub-queries have been answered."""
        return all(q in self.answered for q in self.sub_queries)

    def get_next_target(self):
        """Get the next unanswered sub-query."""
        for q in self.sub_queries:
            if q not in self.answered:
                return q
        return None

    def record_answer(self, sub_query, answer):
        self.answered[sub_query] = answer

    def get_combined_answer(self):
        return " ".join(self.answered.values())
```

### Navigate Back to Root for New Branch
```python
def plan_multi_source_path(root_url, target_pages):
    """Plan navigation paths for multi-source queries.

    Returns a list of paths, each starting from the root URL.
    """
    paths = []
    for target in target_pages:
        paths.append({
            "start": root_url,
            "target": target["description"],
            "keywords": target["keywords"]
        })
    return paths
```

## Action Counting and Budgeting

### Manage Action Budget
```python
class ActionBudget:
    def __init__(self, max_actions=10):
        self.max_actions = max_actions
        self.actions_taken = 0

    def take_action(self):
        self.actions_taken += 1
        return self.actions_taken <= self.max_actions

    def remaining(self):
        return max(0, self.max_actions - self.actions_taken)

    def should_shortcut(self):
        """Check if budget is running low and agent should try to answer."""
        return self.remaining() <= 2
```

## Common Tasks

### Select Link by Button Text
```python
import json

def click_button(button_text, button_map_file="button_url_map.json"):
    """Resolve a button text to its URL using the button map."""
    with open(button_map_file, "r") as f:
        button_map = json.load(f)
    clean_text = button_text.replace("<button>", "")
    return button_map.get(clean_text)
```

### Build Sitemap from Navigation
```python
def build_sitemap(root_url, get_links_fn, max_depth=2):
    """Build a simple sitemap by crawling links."""
    from collections import deque
    sitemap = {}
    queue = deque([(root_url, 0)])
    visited = set()

    while queue:
        url, depth = queue.popleft()
        if url in visited or depth > max_depth:
            continue
        visited.add(url)
        links = get_links_fn(url)
        sitemap[url] = [link["text"] for link in links]
        for link in links:
            if link["url"] not in visited:
                queue.append((link["url"], depth + 1))

    return sitemap
```

### Decide When to Stop Navigating
```python
def should_stop_navigating(memory, query, max_steps, current_step):
    """Decide whether the agent should stop and answer."""
    # Out of budget
    if current_step >= max_steps:
        return True
    # All sub-questions answered (heuristic check)
    if len(memory) >= 2 and "and" in query.lower():
        return True
    # Single-source and found info
    if len(memory) >= 1 and "and" not in query.lower():
        return True
    return False
```

## Quick Reference

| Task | Strategy | When to Use |
|------|----------|-------------|
| Single fact lookup | Follow most relevant link | Single-source easy questions |
| Multi-part query | Visit multiple branches from root | Multi-source questions |
| Deep information | Follow chain of links depth-first | Hard questions requiring deep navigation |
| Broad exploration | BFS from root | When unsure which page has the answer |
| Dead end recovery | Backtrack to parent/root | When current page has no useful info |
| Loop avoidance | Track visited URLs | Always |
| Budget management | Count actions, shortcut when low | Always |
