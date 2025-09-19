"""
Script to auto-generate tests/doc_examples_test.py from README.md and docs/source/examples.rst.
Extracts all Python code blocks and wraps them as pytest-compatible test functions.
"""

import re
from pathlib import Path


def extract_code_blocks(text, file_label):
    # Matches ```python ... ``` or .. code-block:: python ...
    code_blocks = []
    # Markdown style
    md_pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    for match in md_pattern.finditer(text):
        code = match.group(1).strip()
        code_blocks.append((file_label, code))

    # reST style: find each .. code-block:: python and extract the indented block after it
    rst_lines = text.splitlines()
    i = 0
    while i < len(rst_lines):
        line = rst_lines[i]
        if line.strip().startswith(".. code-block:: python"):
            # Find the start of the indented block (skip possible blank lines)
            i += 1
            # Skip blank lines
            while i < len(rst_lines) and rst_lines[i].strip() == "":
                i += 1
            # Collect indented lines
            block_lines = []
            while i < len(rst_lines):
                ell = rst_lines[i]
                if ell.startswith("    "):
                    block_lines.append(ell[4:])
                    i += 1
                elif ell.strip() == "":
                    block_lines.append("")
                    i += 1
                else:
                    break
            code = "\n".join(block_lines).rstrip()
            if code:
                code_blocks.append((file_label, code))
        else:
            i += 1
    return code_blocks


def main():
    base = Path(__file__).parent.parent
    readme = base / "README.md"
    examples = base / "docs" / "source" / "examples.rst"
    out_file = base / "tests" / "doc_examples_test.py"

    code_blocks = []
    if readme.exists():
        code_blocks += extract_code_blocks(readme.read_text(), "README")
    if examples.exists():
        code_blocks += extract_code_blocks(examples.read_text(), "EXAMPLES")

    header = (
        "# Auto-generated from README.md and examples.rst\n"
        "# Each function corresponds to a code block example from the documentation.\n"
        "# These are intended for pytest discovery and basic execution.\n\n"
        "import torch\n"
        "import matplotlib.pyplot as plt\n\n"
    )
    lines = [header]
    for i, (label, code) in enumerate(code_blocks, 1):
        func_name = f"test_{label.lower()}_example_{i}"
        lines.append(f"def {func_name}():\n")
        # Indent code
        for line in code.splitlines():
            lines.append(f"    {line}")
        lines.append("")
    out_file.write_text("\n".join(lines))
    print(f"Wrote {len(code_blocks)} test functions to {out_file}")


if __name__ == "__main__":
    main()
