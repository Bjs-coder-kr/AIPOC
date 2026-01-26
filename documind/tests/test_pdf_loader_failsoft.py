"""PDF loader fail-soft tests."""

import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from documind.ingest import pdf_loader


class _FakePage:
    def __init__(self, text: str | None = None, fail: bool = False) -> None:
        self._text = text
        self._fail = fail

    def extract_text(self) -> str | None:
        if self._fail:
            raise KeyError("bbox")
        return self._text


class _FakeReader:
    def __init__(self, *_args, **_kwargs) -> None:
        self.pages = [
            _FakePage(text="OK page"),
            _FakePage(fail=True),
        ]


def test_load_pdf_failsoft(monkeypatch, caplog) -> None:
    monkeypatch.setattr(pdf_loader, "PdfReader", _FakeReader)
    caplog.set_level(logging.WARNING)

    result = pdf_loader.load_pdf(b"dummy", "sample.pdf")

    assert len(result["pages"]) == 2
    assert result["pages"][0]["text"] == "OK page"
    assert result["pages"][1]["text"] == ""
    assert any("extract_text_failed" in record.message for record in caplog.records)

