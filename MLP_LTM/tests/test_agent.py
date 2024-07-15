import pytest
from unittest.mock import AsyncMock, patch
from app.agent import Agent
from app.db.memory_manager import MemoryManager

@pytest.fixture
def agent():
    memory_manager = AsyncMock(spec=MemoryManager)
    memory_manager.get_relevant_memories.return_value = [
        ("Memory 1 content", 0.9),
        ("Memory 2 content", 0.8)
    ]
    return Agent("fake_api_key", memory_manager)

@pytest.mark.asyncio
async def test_process_query(agent):
    with patch.object(agent.together_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
        mock_create.return_value.choices[0].message.content = "Mocked response"
        
        response = await agent.process_query("What is the capital of France?")
        
        assert response == "Mocked response"
        agent.memory_manager.get_relevant_memories.assert_called_once()
        agent.memory_manager.add_memory.assert_called_once()
        mock_create.assert_called_once()