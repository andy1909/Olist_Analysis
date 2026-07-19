# tests/unit/utils/test_logger.py
"""
Unit tests for src/utils/logger.py

Covers:
    - setup_logger : logger name, handler count, log file creation, idempotency
"""
import logging
import os
import shutil
import tempfile
import unittest

from src.utils.logger import setup_logger

# Logger name as defined in the module under test
_LOGGER_NAME = 'olist_analysis'


class TestSetupLogger(unittest.TestCase):
    """Tests for setup_logger()."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove all handlers to avoid cross-test pollution
        logger = logging.getLogger(_LOGGER_NAME)
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        shutil.rmtree(self.tmp_dir)

    def test_returns_logger_with_correct_name(self):
        logger = setup_logger(log_dir=self.tmp_dir)
        self.assertEqual(logger.name, _LOGGER_NAME)

    def test_logger_level_is_info(self):
        logger = setup_logger(log_dir=self.tmp_dir)
        self.assertEqual(logger.level, logging.INFO)

    def test_attaches_exactly_two_handlers(self):
        """Logger should have one FileHandler and one StreamHandler."""
        logger = setup_logger(log_dir=self.tmp_dir)
        self.assertEqual(len(logger.handlers), 2)

    def test_one_of_the_handlers_is_a_file_handler(self):
        logger = setup_logger(log_dir=self.tmp_dir)
        handler_types = [type(h) for h in logger.handlers]
        self.assertIn(logging.FileHandler, handler_types)

    def test_one_of_the_handlers_is_a_stream_handler(self):
        logger = setup_logger(log_dir=self.tmp_dir)
        # FileHandler is a subclass of StreamHandler; use exact type check
        stream_handlers = [
            h for h in logger.handlers
            if type(h) is logging.StreamHandler
        ]
        self.assertEqual(len(stream_handlers), 1)

    def test_creates_log_file_inside_specified_directory(self):
        setup_logger(log_dir=self.tmp_dir)
        log_files = [f for f in os.listdir(self.tmp_dir) if f.endswith('.log')]
        self.assertEqual(len(log_files), 1,
                         msg="Expected exactly one .log file to be created.")

    def test_log_filename_follows_timestamp_format(self):
        """Expected pattern: YYYY_MM_DD_HH_MM_SS.log"""
        import re
        setup_logger(log_dir=self.tmp_dir)
        log_files = os.listdir(self.tmp_dir)
        pattern = re.compile(r'^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}\.log$')
        self.assertTrue(
            any(pattern.match(f) for f in log_files),
            msg=f"No log file matching timestamp pattern found. Files: {log_files}"
        )

    def test_creates_log_directory_if_it_does_not_exist(self):
        new_log_dir = os.path.join(self.tmp_dir, 'auto_created_logs')
        self.assertFalse(os.path.exists(new_log_dir))

        setup_logger(log_dir=new_log_dir)

        self.assertTrue(os.path.exists(new_log_dir))

    def test_second_call_does_not_duplicate_handlers(self):
        """Guard against the duplicate-handler bug (logger already has handlers)."""
        setup_logger(log_dir=self.tmp_dir)
        setup_logger(log_dir=self.tmp_dir)   # second call

        logger = logging.getLogger(_LOGGER_NAME)
        self.assertEqual(len(logger.handlers), 2,
                         msg="Handlers were duplicated on second setup_logger() call.")

    def test_returned_logger_can_emit_info_message_without_raising(self):
        logger = setup_logger(log_dir=self.tmp_dir)
        try:
            logger.info("Unit test log message")
        except Exception as e:
            self.fail(f"logger.info() raised unexpectedly: {e}")


if __name__ == '__main__':
    unittest.main()
