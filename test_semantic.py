import unittest
from semantic_checks import run_semantic_tests


class SemanticSuite(unittest.TestCase):
    def test_semantic_pipeline(self):
        result = run_semantic_tests()
        self.assertTrue(result['passed'], result)
