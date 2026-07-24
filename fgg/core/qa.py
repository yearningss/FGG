"""
Automated Quality Assurance (QA) Engine for Game Localization.
Validates translation outputs against missing placeholders, broken tags, and bracket mismatches.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple
from fgg.core.placeholders import PlaceholderEngine


@dataclass
class QAIssue:
    key: str
    issue_type: str  # 'missing_tag', 'bracket_mismatch', 'empty_value', 'suspicious_length'
    message: str
    original: str
    translated: str


@dataclass
class QAReport:
    total_checked: int = 0
    issues: List[QAIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.issues) == 0


class QAValidator:
    def __init__(self, placeholder_engine: PlaceholderEngine | None = None) -> None:
        self.ph_engine = placeholder_engine or PlaceholderEngine()

    def validate(self, original_entries: List[Tuple[str, str, str]]) -> QAReport:
        """
        Validates list of tuples: (key, original_text, translated_text)
        Returns a QAReport.
        """
        report = QAReport(total_checked=len(original_entries))

        for key, orig, trans in original_entries:
            if not trans.strip():
                report.issues.append(
                    QAIssue(
                        key=key,
                        issue_type="empty_value",
                        message="Translated text is empty",
                        original=orig,
                        translated=trans,
                    )
                )
                continue

            # 1. Check placeholders & tags
            _, orig_ph = self.ph_engine.extract(orig)
            _, trans_ph = self.ph_engine.extract(trans)

            missing = set(orig_ph) - set(trans_ph)
            if missing:
                report.issues.append(
                    QAIssue(
                        key=key,
                        issue_type="missing_tag",
                        message=f"Missing tags/placeholders: {', '.join(missing)}",
                        original=orig,
                        translated=trans,
                    )
                )

            # 2. Check bracket counts
            for b_open, b_close in [("[", "]"), ("{", "}"), ("<", ">")]:
                if orig.count(b_open) != trans.count(b_open) or orig.count(b_close) != trans.count(b_close):
                    report.issues.append(
                        QAIssue(
                            key=key,
                            issue_type="bracket_mismatch",
                            message=f"Mismatch in brackets ({b_open}{b_close})",
                            original=orig,
                            translated=trans,
                        )
                    )
                    break

        return report
