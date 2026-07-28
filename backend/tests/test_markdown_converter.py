"""
Unit tests for Markdown conversion.
"""
import pytest
from backend.ingest import MarkdownConverter


def test_markdown_converter_basic():
    """Test basic Markdown conversion."""
    converter = MarkdownConverter()
    
    # Test heading detection
    text = "INTRODUCTION\nThis is some text."
    result = converter.convert(text)
    assert "##" in result or "###" in result
    
    # Test list detection
    text = "- Item 1\n- Item 2"
    result = converter.convert(text)
    assert "- Item 1" in result
    assert "- Item 2" in result
    
    # Test numbered list
    text = "1. First\n2. Second"
    result = converter.convert(text)
    assert "1." in result
    assert "2." in result


def test_markdown_converter_preserves_structure():
    """Test that structure is preserved."""
    converter = MarkdownConverter()
    
    text = """METHODS
This study uses the following approach:
- Data collection
- Analysis
- Results"""
    
    result = converter.convert(text)
    
    # Should have heading
    assert "##" in result or "###" in result
    # Should have list items
    assert "- Data collection" in result
