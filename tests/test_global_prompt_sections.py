"""Standalone check for verbatim global sections without loading ComfyUI."""

import ast
import os
import re


def main():
    """Exercise replacements, missing sections, literal escapes and absent globals."""
    path = os.path.join(os.path.dirname(__file__), "..", "nodes.py")
    with open(path) as handle:
        tree = ast.parse(handle.read())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_preserve_global_prompt_sections")
    scope = {"re": re}
    exec(compile(ast.Module(body=[function], type_ignores=[]), path, "exec"), scope)
    preserve = scope[function.name]
    subjects = "subject_definitions:\n  Keep <Subject 1> and literal \\1.\n\n"
    retention = "retention_analysis: Keep exact spacing.  \n\n"
    original = subjects + "summary: Original\n" + retention + "detailed_description: Original dialogue\n"
    chunk = "subject_definitions: Changed\nsummary: Continuation\nretention_analysis: Changed\ndetailed_description: Timed dialogue\n"
    expected = subjects + "summary: Original\n" + retention + "detailed_description: Timed dialogue\n"
    assert preserve(chunk, original) == expected
    restored = preserve("detailed_description: Timed dialogue\n", original)
    assert restored == expected
    assert preserve(restored, original) == restored
    assert preserve(chunk, "summary: Global\n") == "summary: Global\ndetailed_description: Timed dialogue\n"
    print("Global prompt section checks passed.")


if __name__ == "__main__":
    main()
