"""The deterministic verifier: each tool's checks live in its own module, on contract instances."""

from contract import Task, Tool

from . import describe, exact, rank, select, top

TOOLS = {
    Tool.DESCRIBE_COLLECTION: describe,
    Tool.SELECT_PAPERS: select,
    Tool.RANK_TERMS: rank,
    Tool.EXACT_TOP_TERMS: exact,
}
# task -> its output's check, verify_task(arguments, output, reference, workspace)
TASKS = {Task.TOP_TFIDF_TERMS: top.verify_task}


def verify_input(tool, value):
    return TOOLS[Tool(tool)].verify_input(value)


def verify_output(tool, input_value, output_value, workspace):
    return TOOLS[Tool(tool)].verify_output(input_value, output_value, workspace)


def verify_task(task, arguments, output, reference, workspace):
    return TASKS[Task(task)](arguments, output, reference, workspace)
