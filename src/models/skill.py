from __future__ import annotations

import re
from pathlib import Path


class Skill:
    def __init__(self, name: str, description: str, detail: str):
        self.name = name
        self.description = description
        self.detail = detail

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "detail": self.detail,
        }

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @classmethod
    def from_markdown(cls, path: str | Path) -> "Skill":
        """Parse a SKILL.md file and return a populated Skill instance.

        Expected file format::

            ---
            name: <skill_name>
            description: >
              <multiline description …>
            ---

            # <markdown body used as detail>

        Args:
            path: Absolute or relative path to a SKILL.md file.

        Returns:
            A :class:`Skill` with ``name``, ``description``, and ``detail``
            extracted from the file.

        Raises:
            ValueError: If the file does not contain a valid YAML frontmatter
                block (delimited by ``---``).
        """
        content = Path(path).read_text(encoding="utf-8")

        # Split on the YAML frontmatter delimiters (---)
        # Pattern: optional leading whitespace, then ---\n…\n---
        frontmatter_pattern = re.compile(
            r"^\s*---\s*\n(.*?)\n---\s*\n?(.*)",
            re.DOTALL,
        )
        match = frontmatter_pattern.match(content)
        if not match:
            raise ValueError(
                f"No valid YAML frontmatter found in '{path}'. "
                "Expected the file to start with a '---' block."
            )

        raw_frontmatter, raw_body = match.group(1), match.group(2)

        name = cls._extract_frontmatter_field(raw_frontmatter, "name")
        description = cls._extract_frontmatter_field(raw_frontmatter, "description")
        detail = raw_body.strip()

        return cls(name=name, description=description, detail=detail)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_frontmatter_field(frontmatter: str, field: str) -> str:
        """Extract a scalar or block-scalar value for *field* from raw YAML text.

        Supports two common YAML styles used in SKILL.md files:

        * **Scalar** — ``name: daily_planner``
        * **Block scalar** (``>`` / ``|``) — multiline value indented on the
          following lines.

        Args:
            frontmatter: The raw text between the ``---`` delimiters.
            field: The YAML key to extract.

        Returns:
            The extracted value as a stripped string.

        Raises:
            ValueError: If *field* is not found in *frontmatter*.
        """
        lines = frontmatter.splitlines()

        for i, line in enumerate(lines):
            # Match "field: value" or "field: >" / "field: |"
            key_pattern = re.compile(rf"^{re.escape(field)}\s*:\s*(.*)")
            m = key_pattern.match(line.strip())
            if m is None:
                continue

            inline_value = m.group(1).strip()

            # Block scalar indicator — collect indented continuation lines
            if inline_value in (">", "|", ""):
                collected: list[str] = []
                for continuation in lines[i + 1 :]:
                    # Continuation lines are indented
                    if continuation and not continuation[0].isspace():
                        break
                    collected.append(continuation.strip())
                return " ".join(collected).strip()

            return inline_value

        raise ValueError(
            f"Field '{field}' not found in frontmatter:\n{frontmatter}"
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return f"Skill(name={self.name!r})"