"""Real LangChain/LangGraph orchestration with a deterministic local chat model."""

import re
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import ToolException
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import Field

from langchain_tavily import TavilySearch
from tests.unit_tests.test_error_contract import CASES, KEY, invoke, transport


class ScriptedModel(BaseChatModel):
    """Exercise the actual agent loop, without an LLM provider or API key."""

    tool_name: str = "tavily_search"
    arguments: dict[str, Any] = Field(
        default_factory=lambda: {"query": "public example"}
    )

    @property
    def _llm_type(self) -> str:
        return "test-scripted-tool-caller"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        assert any(tool.name == self.tool_name for tool in tools)
        return self

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        reply = (
            AIMessage(content="Finished after observing the tool result")
            if any(isinstance(m, ToolMessage) for m in messages)
            else AIMessage(
                content="",
                tool_calls=[
                    {"name": self.tool_name, "args": self.arguments, "id": "agent-call"}
                ],
            )
        )
        return ChatResult(generations=[ChatGeneration(message=reply)])


@pytest.mark.parametrize("cls,args,success", CASES)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("handled", [False, True])
async def test_tool_node_propagates_or_marks_error(
    cls, args, success, asynchronous, handled
):
    tool = cls(tavily_api_key=KEY, handle_tool_error=handled)
    graph = StateGraph(MessagesState)
    graph.add_node("tools", ToolNode([tool]))
    graph.set_entry_point("tools")
    graph.set_finish_point("tools")
    node = graph.compile()
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": tool.name, "args": args, "id": "node-call"}],
            )
        ]
    }
    with transport(432, {"detail": "credit limit exceeded"}):
        if not handled:
            with pytest.raises(ToolException, match="432.*credit limit"):
                await invoke(node, state, asynchronous)
        else:
            result = await invoke(node, state, asynchronous)
            message = result["messages"][-1]
            assert message.status == "error" and message.tool_call_id == "node-call"
            assert "432" in message.content


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("handled", [False, True])
async def test_create_agent_http_error_contract(asynchronous, handled):
    tool = TavilySearch(tavily_api_key=KEY, handle_tool_error=handled)
    agent = create_agent(ScriptedModel(), [tool])
    state = {"messages": [{"role": "user", "content": "Find the example"}]}
    with transport(401, {"detail": {"error": "invalid key"}}):
        if not handled:
            with pytest.raises(ToolException, match="401.*invalid key"):
                await invoke(agent, state, asynchronous)
        else:
            result = await invoke(agent, state, asynchronous)
            messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
            assert len(messages) == 1 and messages[0].status == "error"
            assert result["messages"][-1].content.startswith("Finished")


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_create_agent_forbidden_parameter_is_not_success(asynchronous):
    tool = TavilySearch(tavily_api_key=KEY, max_results=5)
    model = ScriptedModel(arguments={"query": "test", "max_results": 5})
    agent = create_agent(model, [tool])
    with transport() as (post, _, session), pytest.raises(
        ValueError, match="only be set during instantiation"
    ):
        await invoke(
            agent, {"messages": [{"role": "user", "content": "test"}]}, asynchronous
        )
    post.assert_not_called()
    session.post.assert_not_called()


def test_readme_agent_example_executes_actual_block(monkeypatch):
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text()
    section = readme.split("### Agent Tool Calling", 1)[1].split(
        "## Tavily Extract", 1
    )[0]
    code = re.search(r"```python\n(.*?)```", section, re.DOTALL).group(1)
    monkeypatch.setenv("TAVILY_API_KEY", KEY)
    with patch(
        "langchain.chat_models.init_chat_model", return_value=ScriptedModel()
    ), transport(200, CASES[0][2]):
        namespace: dict[str, Any] = {}
        exec(compile(code, "README-agent-example", "exec"), namespace)
    response = namespace["response"]
    messages = [m for m in response["messages"] if isinstance(m, ToolMessage)]
    assert len(messages) == 1 and messages[0].status == "success"
    assert response["messages"][-1].content.startswith("Finished")
