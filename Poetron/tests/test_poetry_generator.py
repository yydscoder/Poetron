"""
Basic tests for the Poetry Generator (AI GENERATED)
"""

import unittest
import tempfile
import os
from pathlib import Path

# Add the src directory to the path so we can import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils import export_poem, validate_style, format_poem_for_style
from data_preprocessing import clean_poem_text, add_style_tokens
from poetry_generator import generate_fallback_poem


class TestUtils(unittest.TestCase):
    
    def test_export_poem(self):
        """Test that poems can be exported to files."""
        poem = "This is a test poem.\nWith multiple lines."
        style = "freeverse"
        
        filename = export_poem(poem, style)
        
        # Check that file was created
        self.assertTrue(os.path.exists(filename))
        
        # Check that file contains expected content
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("Style: freeverse", content)
            self.assertIn(poem, content)
        
        # Clean up
        os.remove(filename)
    
    def test_validate_style(self):
        """Test style validation."""
        self.assertTrue(validate_style('haiku'))
        self.assertTrue(validate_style('sonnet'))
        self.assertTrue(validate_style('freeverse'))
        self.assertFalse(validate_style('invalid_style'))
    
    def test_format_poem_for_style(self):
        """Test poem formatting for different styles."""
        test_poem = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        
        # Test haiku formatting (should limit to 3 lines)
        formatted_haiku = format_poem_for_style(test_poem, 'haiku')
        self.assertEqual(len(formatted_haiku.split('\n')), 3)
        
        # Test sonnet formatting (should limit to 14 lines, but we only have 5)
        formatted_sonnet = format_poem_for_style(test_poem, 'sonnet')
        self.assertEqual(len(formatted_sonnet.split('\n')), 5)
        
        # Test freeverse formatting (should remain unchanged)
        formatted_freeverse = format_poem_for_style(test_poem, 'freeverse')
        self.assertEqual(formatted_freeverse, test_poem)


class TestDataPreprocessing(unittest.TestCase):
    
    def test_clean_poem_text(self):
        """Test poem text cleaning."""
        dirty_text = "  This   is  \t a  \n\n  test   \r\n  poem.  "
        cleaned = clean_poem_text(dirty_text)
        
        # Should remove extra whitespace
        self.assertNotIn("  ", cleaned)
        # Should normalize line breaks
        self.assertEqual(cleaned.count('\n'), 1)  # Only one newline between sentences
        # Should strip leading/trailing whitespace
        self.assertEqual(cleaned, "This is a test poem.")
    
    def test_add_style_tokens(self):
        """Test adding style tokens to poems."""
        poems = ["Poem 1", "Poem 2"]
        style = "haiku"
        
        result = add_style_tokens(poems, style)
        
        self.assertEqual(result[0], "<HAIKU> Poem 1")
        self.assertEqual(result[1], "<HAIKU> Poem 2")


class TestPoetryGenerator(unittest.TestCase):
    
    def test_generate_fallback_poem(self):
        """Test fallback poem generation."""
        haiku = generate_fallback_poem('haiku', 'moon')
        self.assertIsNotNone(haiku)
        self.assertLessEqual(len(haiku.split('\n')), 3)  # Haiku should be 3 lines or less
        
        sonnet = generate_fallback_poem('sonnet', 'love')
        self.assertIsNotNone(sonnet)
        self.assertLessEqual(len(sonnet.split('\n')), 14)  # Sonnet should be 14 lines or less
        
        freeverse = generate_fallback_poem('freeverse', 'ocean')
        self.assertIsNotNone(freeverse)


if __name__ == '__main__':
    unittest.main()