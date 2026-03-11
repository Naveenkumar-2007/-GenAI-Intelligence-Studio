"""Code explanation tool using the main LLM."""

from langchain_core.tools import Tool
from langchain_core.language_models import BaseLanguageModel


def build_code_explainer_tool(llm: BaseLanguageModel) -> Tool:
    """
    Build a Tool that explains code using the LLM.
    """

    def _explain_code(code: str) -> str:
        prompt = (
            "You are a senior software engineer. Explain the following code clearly:\n\n"
            f"```python\n{code}\n```"
        )
        resp = llm.invoke(prompt)
        return resp.content if hasattr(resp, "content") else str(resp)

    return Tool(
        name="code_explainer",
        description="Explain Python / general source code in simple terms.",
        func=_explain_code,
    )
