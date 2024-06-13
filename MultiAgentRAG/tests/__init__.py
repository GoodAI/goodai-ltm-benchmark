import unittest
from src.utils.data_utils import load_and_process_data
from src.memory.memory_manager import MemoryManager

class TestDataUtils(unittest.TestCase):
    def test_load_and_process_data(self):
        # Assuming there's a test PDF in data/raw for testing
        docs = load_and_process_data("data/raw")
        self.assertTrue(len(docs) > 0, "Should load and split documents")

class TestMemoryManager(unittest.TestCase):
    def setUp(self):
        self.memory_manager = MemoryManager(":memory:")  # Use in-memory database for testing

    def test_save_and_get_memories(self):
        self.memory_manager.save_memory("test_query", "test_result")
        memories = self.memory_manager.get_memories()
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0][0], "test_query")
        self.assertEqual(memories[0][1], "test_result")

if __name__ == "__main__":
    unittest.main()
