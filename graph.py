# agent/graph.py
# -------------------------
# Compatibility shim + main graph
# ----------------------a--- 
from dotenv import load_dotenv
load_dotenv()

# --- Compatibility shim: gracefully import set_debug / set_verbose ---
try:
    from langchain.globals import set_verbose, set_debug
except Exception:
    try:
        from langchain_core.globals import set_verbose, set_debug
    except Exception:
        # fallback no-ops so file doesn't crash during import
        def set_debug(val: bool):
            return None

        def set_verbose(val: bool):
            return None

# --- Compatibility shim: create_react_agent wrapper ---
def _make_compat_create_react_agent():
    try:
        # older / expected location
        from langgraph.prebuilt import create_react_agent as _c
        return _c
    except Exception:
        try:
            # fallback to langchain's create_agent (signature may differ by version)
            from langchain.agents import create_agent as _create_agent

            # Wrap create_agent to accept the older (llm, tools) signature.
            def _wrapper(llm, tools, *args, **kwargs):
                # Many langchain.create_agent implementations expect llm and tools=...
                return _create_agent(llm, tools=tools, *args, **kwargs)

            return _wrapper
        except Exception:
            def _no_op(*args, **kwargs):
                raise ImportError(
                    "Neither langgraph.prebuilt.create_react_agent nor langchain.agents.create_agent is available. "
                    "Please install or update langgraph/langchain."
                )
            return _no_op

create_react_agent = _make_compat_create_react_agent()

# -------------------------
# Your normal imports and code
# -------------------------
from typing import Dict

# LLM and framework imports
try:
    from langchain_groq.chat_models import ChatGroq
except Exception:
    # If the explicit subpackage isn't present, try a more direct import
    try:
        from langchain_groq import ChatGroq  # type: ignore
    except Exception as e:
        raise ImportError("ChatGroq import failed. Install langchain-groq (pip install langchain-groq).") from e

from langgraph.constants import END
from langgraph.graph import StateGraph

# local project modules
from agent.prompts import *
from agent.states import *
from agent.tools import write_file, read_file, get_current_directory, list_files

# enable verbose/debug if available
set_debug(True)
set_verbose(True)

# initialize the model (edit model name if needed)
llm = ChatGroq(model="openai/gpt-oss-120b")


def planner_agent(state: dict) -> dict:
    """Converts user prompt into a structured Plan."""
    user_prompt = state.get("user_prompt", "")
    if not user_prompt:
        raise ValueError("No user_prompt provided to planner_agent.")
    # Use structured_output typing Plan (from agent.states)
    resp = llm.with_structured_output(Plan).invoke(
        planner_prompt(user_prompt)
    )
    if resp is None:
        raise ValueError("Planner did not return a valid response.")
    return {"plan": resp}


def architect_agent(state: dict) -> dict:
    """Creates TaskPlan from Plan."""
    plan: Plan = state.get("plan")
    if plan is None:
        raise ValueError("architect_agent expected 'plan' in state.")
    resp = llm.with_structured_output(TaskPlan).invoke(
        architect_prompt(plan=plan.model_dump_json())
    )
    if resp is None:
        raise ValueError("Architect did not return a valid response.")

    # attach the original plan to the returned task plan for context
    try:
        resp.plan = plan
    except Exception:
        # If resp is a primitive or missing attribute, ignore but continue
        pass

    # Print for debugging
    try:
        print(resp.model_dump_json())
    except Exception:
        # safe fallback
        print("Architect response:", resp)

    return {"task_plan": resp}


def coder_agent(state: dict) -> dict:
    """LangGraph tool-using coder agent."""
    coder_state: CoderState = state.get("coder_state")
    if coder_state is None:
        # first invocation: create initial coder state from task_plan
        coder_state = CoderState(task_plan=state["task_plan"], current_step_idx=0)

    steps = coder_state.task_plan.implementation_steps
    if coder_state.current_step_idx >= len(steps):
        return {"coder_state": coder_state, "status": "DONE"}

    current_task = steps[coder_state.current_step_idx]
    # read existing content via your tool wrapper (read_file.run)
    try:
        existing_content = read_file.run(current_task.filepath)
    except Exception:
        existing_content = ""

    system_prompt = coder_system_prompt()
    user_prompt = (
        f"Task: {getattr(current_task, 'task_description', current_task)}\n"
        f"File: {getattr(current_task, 'filepath', 'unknown')}\n"
        f"Existing content:\n{existing_content}\n"
        "Use write_file(path, content) to save your changes."
    )

    # Provide the tools to the agent. create_react_agent is shimmed above.
    coder_tools = [read_file, write_file, list_files, get_current_directory]
    react_agent = create_react_agent(llm, coder_tools)

    # invoke react agent with messages
    try:
        react_agent.invoke({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        })
    except Exception as e:
        # If the agent invocation fails, surface an informative error and continue
        print("React agent invocation failed:", e)

    coder_state.current_step_idx += 1
    return {"coder_state": coder_state}


# Build graph
graph = StateGraph(dict)

graph.add_node("planner", planner_agent)
graph.add_node("architect", architect_agent)
graph.add_node("coder", coder_agent)

graph.add_edge("planner", "architect")
graph.add_edge("architect", "coder")
graph.add_conditional_edges(
    "coder",
    lambda s: "END" if s.get("status") == "DONE" else "coder",
    {"END": END, "coder": "coder"}
)

graph.set_entry_point("planner")
agent = graph.compile()

if __name__ == "__main__":
    # Example invocation — adjust recursion_limit as needed
    try:
        result = agent.invoke({"user_prompt": "Create a simple calculator web application"},
                              {"recursion_limit": 100})
        print("Final State:", result)
    except Exception as e:
        print("Error running agent:", e)
