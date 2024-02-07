import codecs
import os.path
import unittest
from typing import Any

from utils.json_helper import sanitize_and_parse_json


class TestJsonHelper(unittest.TestCase):
    def test_sanitize_and_parse_1(self):
        obj = self._parse_file("test1.txt")
        self.assertIsInstance(obj, dict)
        self.assertIsNotNone(obj.get("user"))
        self.assertIsNotNone(obj.get("queries"))
        self.assertIsNotNone(obj.get("summary"))

    def test_sanitize_and_parse_2(self):
        obj = self._parse_file("test2.txt")
        self.assertEquals("y", obj.get("x"))

    def test_sanitize_and_parse_3(self):
        obj = self._parse_file("test3.txt")
        self.assertEquals("y", obj.get("x"))

    def test_sanitize_and_parse_4(self):
        obj = self._parse_file("test4.txt")
        self.assertEquals("y", obj.get("x"))

    def test_sanitize_and_parse_5(self):
        obj = self._parse_file("test5.txt")
        self.assertEquals("foo", obj.get("x"))
        self.assertEquals("bar", obj.get("y"))

    def test_sanitize_and_parse_6(self):
        obj = self._parse_file("test6.txt")
        self.assertEquals(["foo", "bar"], obj.get("x"))

    def test_sanitize_and_parse_7(self):
        obj = self._parse_file("test7.txt")
        self.assertEquals({"name": "Mikayla"}, obj.get("user"))

    def test_sanitize_and_parse_8(self):
        obj = self._parse_file("test8.txt")
        self.assertIsNotNone(obj.get("user"))
        self.assertIsNotNone(obj.get("queries"))
        self.assertIsNotNone(obj.get("summary"))

    def test_sanitize_and_parse_9(self):
        obj = self._parse_file("test9.txt")
        self.assertIsNotNone(obj.get("user"))
        self.assertIsNotNone(obj.get("queries"))
        self.assertIsNotNone(obj.get("summary"))

    def test_sanitize_and_parse_10(self):
        obj = self._parse_file("test10.txt")
        self.assertIsNotNone(obj.get("user"))
        self.assertIsNotNone(obj.get("queries"))
        self.assertIsNotNone(obj.get("summary"))

    def test_sanitize_and_parse_11(self):
        obj = self._parse_file("test11.txt")
        self.assertIsNotNone(obj.get("user"))
        self.assertIsNotNone(obj.get("queries"))
        self.assertIsNotNone(obj.get("summary"))

    def test_sanitize_and_parse_12(self):
        obj = self._parse_file("test12.txt")
        assert "a" in obj
        assert obj.get("a") is None

    @staticmethod
    def _parse_file(file_name: str) -> Any:
        file_path = os.path.join(os.path.dirname(__file__), 'inputs', file_name)
        with codecs.open(file_path, "r") as fd:
            return sanitize_and_parse_json(fd.read())
