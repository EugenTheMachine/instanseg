# Technical Code Quality Guide

This document outlines the code quality, linting, formatting, and static analysis tools used in the `instanseg` repository, along with Command Prompt (CMD) and PowerShell commands for executing quality checks and automated fixes.

---

## Code Quality Modules

1. **Ruff**
   - *Description*: An extremely fast Python linter written in Rust that enforces style guidelines, detects code smells, unused imports, and potential bugs. It replaces multiple traditional linters (Flake8, Pyflakes, pydocstyle) in a single unified tool.

2. **Black**
   - *Description*: The uncompromising Python code formatter that automatically formats code according to PEP 8 standards with deterministic output. It eliminates code style arguments by reformatting entire files consistently.

3. **isort**
   - *Description*: A Python utility that automatically sorts import statements alphabetically and organizes them into logical sections (standard library, third-party libraries, local modules).

4. **mypy**
   - *Description*: An optional static type checker for Python that verifies type annotations across the codebase. It helps catch type mismatches, missing attributes, and potential `NoneType` errors before runtime.

---

## Code Quality Commands

Run these commands from the root directory of the repository in Command Prompt (`cmd.exe`) or PowerShell.

### 1. Run All Checks (Inspection Only)

```cmd
:: Lint code with Ruff
ruff check .

:: Check formatting with Black
black --check .

:: Check import sorting with isort
isort --check-only .

:: Run type checking with mypy
mypy instanseg
```

---

### 2. Auto-Fix and Reformat Code

```cmd
:: Automatically fix lint errors with Ruff
ruff check --fix .

:: Reformat all Python files with Black
black .

:: Sort imports automatically with isort
isort .
```

---

### 3. One-Liner Full Code Quality Check & Auto-Fix

```cmd
isort . && black . && ruff check --fix . && mypy instanseg
```
