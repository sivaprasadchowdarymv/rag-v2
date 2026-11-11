"""Shared prompt text (answer format, not-found marker, vision prompts)."""
from __future__ import annotations

NOT_FOUND = "NOT FOUND IN DOCUMENT"

ANSWER_FORMAT = f"""Write the final answer in Markdown with these sections:

**Answer:** the direct answer in 1-3 sentences: value, unit, and conditions (min/typ/max, temperature, test conditions), with citations.

**Explanation:** what it means in plain engineering language: what the parameter or concept is, why it matters, the conditions it applies to, and any caveats, with citations.

**Equations:** ONLY if an equation or formula is relevant. Write each one in LaTeX display math on its own line as $$ ... $$, then list what every symbol means with its unit, with citations. If you computed a value, show the substituted equation and the result. Omit this whole section when no equation applies.

Rules:
- Use ONLY facts from the numbered evidence / tool observations. Never invent values, symbols or equations.
- Cite every factual statement by copying the exact label printed before the evidence you used (format [REF-n: TYPE pX]). Never make up a label.
- Use $...$ for inline math and $$...$$ for display math (not \\( \\) or \\[ \\]).
- Do the arithmetic with the calculate tool when it is available; never estimate numbers in your head.
- If the document does not contain the answer, reply exactly: {NOT_FOUND}"""

FIGURE_PROMPT = ("You are indexing a figure from a technical document so it can be found by search later. "
                 "Describe it precisely and factually in at most 120 words: the figure type (graph, block "
                 "diagram, schematic, pinout, photo, table image, ...), axis names and units with ranges, "
                 "each curve or block and what it shows, labels, and the key values you can read. "
                 "Do not speculate beyond what is visible.{hint}")

OCR_PROMPT = ("Transcribe ALL text on this scanned document page exactly, in reading order, as Markdown. "
              "Keep headings as Markdown headings, tables as Markdown tables (keep every row, unit and "
              "footnote), and equations as plain text such as PD = (VIN - VOUT) x IOUT. "
              "Do not summarise, translate or add anything. If the page has no text, reply with nothing.")
