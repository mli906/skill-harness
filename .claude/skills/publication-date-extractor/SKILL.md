---
name: publication-date-extractor
description: Extract publication date information from structured texts.
compatibility: opencode
---

This skill focuses on identifying and retrieving specific publication dates from various structured texts, even when datasets are incomplete. It leverages patterns and contextual clues to ensure accuracy in extraction.

### Rule
- Use regex and parsing techniques to locate publication dates in different formats (e.g., MM/DD/YYYY, YYYY-MM-DD, etc.) and extract them for further processing.

### Examples
1. **Input**: "This article was published on 12/15/2022."
   **Output**: "2022-12-15"

2. **Input**: "The report from 2023-10-12 indicates changes in policy."
   **Output**: "2023-10-12"