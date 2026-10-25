#!/usr/bin/env python3
# File: <project>\moe_track\trace\seed_prompts.py
"""Built-in prompts for the domains the dataset does not cover well: general code and general reasoning.

These are neutral, ordinary requests (nothing here is secret or personal). They are the "other half" of the workload next to the
persona and agent prompts built from dataset/<file>, so the recorded routing is not only persona text.
"""
from __future__ import annotations

CODE_SYSTEM = "You are a careful senior software engineer. Give complete, working code and explain only what is needed."

CODE = [
    "Write a Python function that merges overlapping intervals in a list of (start, end) tuples. Include type hints and three test cases.",
    "Implement an LRU cache in C++17 with O(1) get and put using std::list and std::unordered_map. Show a short main() that exercises it.",
    "Write a PowerShell script that finds the 20 largest files under a folder, shows their size in MB, and exports the list to CSV.",
    "Explain what is wrong with this Python code and fix it:\n\ndef average(xs):\n    total = 0\n    for i in range(len(xs) + 1):\n        total += xs[i]\n    return total / len(xs)",
    "Write a SQL query that returns, for each customer, their most recent order and the number of orders they placed in the last 90 days. Assume tables customers(id, name) and orders(id, customer_id, created_at, total).",
    "Write a Rust function that parses a string like \"3d4h12m\" into a total number of seconds, returning a Result with a useful error type.",
    "Create a Python dataclass-based command-line tool using argparse that renames files in a folder by adding a date prefix, with a --dry-run option.",
    "Write a regular expression that matches valid IPv4 addresses and a Python function that uses it. Explain the pattern piece by piece.",
    "Implement binary search three ways in Python: iterative, recursive, and using the bisect module. Discuss the off-by-one pitfalls.",
    "Write a JavaScript function that debounces another function, with a leading-edge option, and show how to test it with fake timers.",
    "Write a bash script that watches a directory for new .log files and gzips any file older than a day, logging what it did.",
    "Design a small Python class hierarchy for a text-adventure game (Room, Item, Player) and write the move/pickup logic with unit tests.",
    "Write a C function that reverses a singly linked list in place and a second one that detects a cycle. Explain the complexity.",
    "Convert this nested loop into a vectorized NumPy expression and explain the speedup:\n\nfor i in range(n):\n    for j in range(m):\n        out[i][j] = a[i] * b[j] + c",
    "Write a Python generator that streams a very large CSV file in chunks of 10,000 rows and yields cleaned dictionaries, skipping malformed lines.",
    "Write a Go HTTP handler that accepts a JSON body, validates two required fields, and returns a structured error response. Include a table-driven test.",
    "Refactor this function to be easier to test and explain each change:\n\ndef process(path):\n    f = open(path)\n    data = f.read()\n    out = []\n    for line in data.split('\\n'):\n        if line and not line.startswith('#'):\n            out.append(line.strip().upper())\n    print(len(out))\n    return out",
    "Write a TypeScript type-safe event emitter class with on, off and emit, where the event names and payload types are checked at compile time.",
    "Explain the difference between a mutex, a semaphore and a condition variable, then write a C++ bounded producer/consumer queue using them.",
    "Write a Python script using sqlite3 that creates a table, inserts rows from a list of dicts in a single transaction, and prints a summary with GROUP BY.",
    "Implement Dijkstra's algorithm in Python with a priority queue, and show how to reconstruct the shortest path from the predecessor map.",
    "Write a unit test suite with pytest for a function slugify(text) covering unicode, whitespace, punctuation and empty input; then implement the function.",
    "Write a Windows batch file and an equivalent PowerShell command that delete all .tmp files older than seven days in a folder tree, with a confirmation prompt.",
    "Explain how Python's asyncio event loop schedules tasks, then write an example that fetches ten URLs concurrently with a concurrency limit of three.",
    "Review this C++ snippet for undefined behaviour and memory problems, then rewrite it safely:\n\nchar* make(int n) {\n    char buf[64];\n    for (int i = 0; i <= n; ++i) buf[i] = 'a' + i % 26;\n    return buf;\n}",
    "Write a small tokenizer and recursive-descent parser in Python for arithmetic expressions with + - * / parentheses and unary minus, returning the evaluated result.",
    "Show how to implement a thread-safe singleton in modern C++ and explain why double-checked locking was historically broken.",
    "Write a Dockerfile and a short docker-compose.yml for a Python FastAPI app with a Postgres database, using a non-root user and a health check.",
    "Write a Python function that computes the edit distance between two strings with dynamic programming, and then optimise its memory use to O(min(n, m)).",
    "Write an INNO Setup-style checklist in plain steps for packaging a Python desktop app: what to freeze, what to include, how to sign the installer. Keep it practical.",
    "Write a Python context manager that times a block of code and logs the elapsed time, both as a class and with contextlib.contextmanager.",
    "Explain what a memory-mapped file is and write a C++ program on Linux that counts the lines of a 5 GB text file using mmap.",
    "Write a git workflow explanation for resolving a merge conflict in a file edited on two branches, with the exact commands, and a note on when to rebase instead.",
    "Write a Python script that reads a JSON Lines file of chat messages and reports, per day, the number of messages and the average length, using only the standard library.",
    "Implement a trie in Python with insert, search, starts_with and a method that returns all words with a given prefix, and analyse the memory cost.",
    "Write a CMake file for a small C++ library with unit tests (using CTest), an installable target, and a compile option for AddressSanitizer.",
    "Write a Python function to validate a Sudoku board and a second to solve it with backtracking and constraint propagation.",
    "Write a PowerShell function that retries a web request up to five times with exponential backoff and jitter, and surfaces the final error clearly.",
    "Explain the CUDA memory hierarchy (registers, shared, L2, global) and write a simple CUDA kernel for a vector add plus the host code to launch it.",
    "Write a Python script that uses the logging module correctly: one logger per module, a rotating file handler, and a JSON formatter for structured logs.",
]

REASONING_SYSTEM = "You are a helpful, precise assistant. Think carefully and show your reasoning briefly before answering."

REASONING = [
    "A train leaves city A at 9:00 travelling at 80 km/h. A second train leaves city B, 420 km away, at 10:00 travelling toward A at 100 km/h. At what time do they meet?",
    "Three boxes are labelled 'apples', 'oranges' and 'mixed', but every label is wrong. You may take one fruit from one box. How do you fix all the labels?",
    "Explain why the sum of the first n odd numbers is n squared, with an intuitive picture and a short induction proof.",
    "A bag contains 5 red and 7 blue marbles. Two are drawn without replacement. What is the probability that they are different colours? Show the working.",
    "Explain the difference between correlation and causation using three concrete examples, and describe how a randomised experiment resolves the ambiguity.",
    "Summarise the main causes of the 2008 financial crisis in five clear points for a general reader, avoiding jargon.",
    "Plan a three-day itinerary for a first visit to Kyoto in November, balancing temples, food and rest. Explain the ordering of the days.",
    "What are the trade-offs between renting and buying a home? Give a structured comparison and say which factors matter most for a ten-year horizon.",
    "Explain how public-key encryption works to a smart teenager, including why it is safe to publish the public key.",
    "Write a short, persuasive paragraph arguing for a four-day work week, then a paragraph presenting the strongest objection to it.",
    "A farmer has 100 m of fencing and wants to enclose a rectangular field against a long straight wall, using fencing on three sides. What dimensions maximise the area?",
    "Explain what a derivative means geometrically and physically, then differentiate f(x) = x^3 * sin(x) step by step.",
    "Compare TCP and UDP: how each works, what each guarantees, and which kinds of applications choose which. End with a table-like summary in plain text.",
    "If all bloops are razzies and some razzies are lazzies, can we conclude that some bloops are lazzies? Explain with a small diagram in words.",
    "Describe how vaccines train the immune system, and why booster doses are sometimes needed, for a reader with no biology background.",
    "Solve: 3x + 2y = 16 and x - y = 1. Then explain a second method and say which one is faster for larger systems.",
    "Explain the Monty Hall problem and why switching doubles the chance of winning. Then simulate the idea in your head for 3 doors versus 100 doors.",
    "Draft a polite but firm email asking a landlord to repair a broken heater, citing the lease and proposing a deadline. Keep it under 200 words.",
    "Explain how a binary heap works and why heap-sort is O(n log n), then say when you would choose it over quicksort.",
    "What is inflation, who is hurt by it, and who benefits? Give a balanced explanation with two numerical examples.",
    "Explain Bayes' theorem with a medical-test example: a disease with 1% prevalence, a test with 95% sensitivity and 90% specificity. What is P(disease | positive)?",
    "Give a step-by-step method for learning a new programming language in 30 days when you already know another one, with weekly goals.",
    "A tank fills in 6 hours with pipe A alone and in 9 hours with pipe B alone, but a drain empties it in 12 hours. With all three open, how long to fill it?",
    "Explain the difference between supervised, unsupervised and reinforcement learning with one everyday analogy for each.",
    "Write a balanced short essay on whether cities should ban cars from their centres. Present both sides fairly before giving a careful conclusion.",
    "Explain how photosynthesis works at the level of a high-school biology course, and why leaves are green.",
    "A password policy requires 8-12 characters from 62 possible symbols. How many possible passwords are there, roughly, and how long would a brute-force attack take at a billion guesses per second?",
    "Describe the main steps of the scientific method using the example of testing whether plants grow faster with music. Point out two ways the experiment could mislead.",
    "Explain what a mutex deadlock is with a two-thread example in plain language and list three standard ways to avoid it.",
    "Prove that the square root of 2 is irrational, then explain where the proof would break for the square root of 4.",
]
