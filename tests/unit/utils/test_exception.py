# tests/unit/utils/test_exception.py
"""
Unit tests for src/utils/exception.py

Covers:
    - CustomException : message formatting, raisability, inheritance
    - error_message_detail : traceback extraction (tested implicitly via CustomException)
"""
import sys
import unittest

from src.utils.exception import CustomException


class TestCustomException(unittest.TestCase):
    """Tests for CustomException."""

    def test_is_subclass_of_builtin_exception(self):
        exc = CustomException("test error")
        self.assertIsInstance(exc, Exception)

    def test_str_representation_contains_error_message(self):
        exc = CustomException("something went wrong")
        self.assertIn("something went wrong", str(exc))

    def test_can_be_raised_and_caught_as_custom_exception(self):
        with self.assertRaises(CustomException):
            raise CustomException("raised in test")

    def test_can_be_caught_as_generic_exception(self):
        """CustomException must also be catchable via 'except Exception'."""
        caught = False
        try:
            raise CustomException("generic catch test")
        except Exception:
            caught = True
        self.assertTrue(caught)

    def test_stores_plain_message_when_no_traceback_provided(self):
        exc = CustomException("plain message")
        self.assertIn("plain message", exc.error_message)

    def test_error_message_includes_filename_when_traceback_is_provided(self):
        """When sys is passed, the message should contain the source file name."""
        try:
            raise ValueError("inner error")
        except ValueError as inner:
            exc = CustomException(inner, sys)
        # The traceback should reference this test file
        self.assertIn('.py', exc.error_message)

    def test_error_message_includes_line_number_when_traceback_is_provided(self):
        try:
            raise ValueError("inner error for line check")
        except ValueError as inner:
            exc = CustomException(inner, sys)
        # Line numbers are wrapped in brackets: 'line [NN]'
        self.assertIn('line [', exc.error_message)

    def test_multiple_instances_are_independent(self):
        exc1 = CustomException("error one")
        exc2 = CustomException("error two")
        self.assertNotEqual(str(exc1), str(exc2))


if __name__ == '__main__':
    unittest.main()
