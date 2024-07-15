import pytest
from app.db.memory_manager import MemoryManager

@pytest.fixture
def memory_manager():
    return MemoryManager("sqlite:///:memory:")

@pytest.mark.asyncio
async def test_add_and_retrieve_memory(memory_manager):
    await memory_manager.add_memory("Test memory content")
    memories = await memory_manager.get_relevant_memories("Test", top_k=1)
    assert len(memories) == 1
    assert memories[0][0] == "Test memory content"

@pytest.mark.asyncio
async def test_memory_linking(memory_manager):
    await memory_manager.add_memory("The capital of France is Paris.")
    await memory_manager.add_memory("Paris is known for the Eiffel Tower.")
    
    memories = await memory_manager.get_relevant_memories("What is the capital of France?", top_k=2)
    assert len(memories) == 2
    assert any("capital of France" in memory[0] for memory in memories)
    assert any("Eiffel Tower" in memory[0] for memory in memories)
