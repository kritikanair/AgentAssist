
from typing import TypedDict, Any

from llm import chat
from langgraph.graph import StateGraph, START, END

from tools import calculate, generate_report
from rag import retrieve_relevant_chunks


class AgentState(TypedDict, total=False):
    question: str
    route: str
    answer: str
    context: str
    sources: list
    trace: list
    collection: Any
    embedding_model: Any




def route_question(state: AgentState):
    """Route requests to the appropriate AgentAssist capability."""
    question = state["question"].strip()
    normalized = question.lower()

    # Calculator requests
    calculation_terms = [
        "calculate", "compute", "evaluate",
        "what is the result", "solve",
    ]

    if any(term in normalized for term in calculation_terms):
        route = "calculator"

    # Report-generation requests
    elif any(
        term in normalized
        for term in [
            "generate a report",
            "write a report",
            "create a report",
            "prepare a report",
            "summarize into a report",
        ]
    ):
        route = "report"

    # Explicit references to uploaded documents
    elif any(
        term in normalized
        for term in [
            "according to the document",
            "according to the pdf",
            "according to the uploaded document",
            "according to the uploaded pdf",
            "from the document",
            "from the pdf",
            "in the uploaded document",
            "in the uploaded pdf",
            "uploaded document",
            "uploaded pdf",
            "my document",
            "my pdf",
            "document says",
            "pdf says",
            "document contains",
            "document cover",
            "document about",
            "main topics covered",
        ]
    ):
        route = "document_search"

    # All other questions use general chat
    else:
        route = "general"

    return {
        "route": route,
        "trace": [f"Intent classified as: {route}"],
        "sources": [],
        "context": "",
    }


def calculator_node(state: AgentState):
    """Extract an arithmetic expression and execute the calculator tool."""
    question = state["question"]

    expression = chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract the arithmetic expression requested by the user. "
                    "Return only the expression using numbers and basic "
                    "operators (+, -, *, /, **, parentheses). "
                    "Do not calculate the answer yourself. "
                    "If no valid expression is present, return INVALID."
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0,
    ).strip()

    if expression.upper() == "INVALID":
        result = "I couldn't identify a valid arithmetic expression."
    else:
        result = calculate(expression)

    return {
        "answer": result,
        "trace": state.get("trace", [])
        + ["Calculator tool executed."],
    }


def retrieve_context(state: AgentState):
    """Retrieve relevant chunks from the active document collection."""
    collection = state.get("collection")
    embedding_model = state.get("embedding_model")

    if collection is None or embedding_model is None:
        return {
            "context": "",
            "sources": [],
            "trace": state.get("trace", [])
            + ["Document retrieval skipped: no indexed documents."],
        }

    matches = retrieve_relevant_chunks(
        state["question"],
        collection,
        embedding_model,
        top_k=4,
    )

    context = "\n\n".join(
        (
            f"[Source: {item['source']}, "
            f"Page: {item['page'] or 'N/A'}]\n"
            f"{item['text']}"
        )
        for item in matches
    )

    return {
        "context": context,
        "sources": matches,
        "trace": state.get("trace", [])
        + [f"Retrieved {len(matches)} document chunks."],
    }



def document_search_node(state: AgentState):
    """Answer from documents when possible, otherwise use general chat."""
    retrieved = retrieve_context(state)
    context = retrieved["context"]

    if not context:
        answer = chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are AgentAssist. Answer general questions "
                        "clearly and accurately. No document context was "
                        "available for this question."
                    ),
                },
                {"role": "user", "content": state["question"]},
            ],
        )

        trace = retrieved["trace"] + [
            "No document context found; answered using general LLM knowledge."
        ]

        return {
            "answer": answer,
            "context": "",
            "sources": [],
            "trace": trace,
        }

    answer = chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are AgentAssist. First determine whether the "
                    "retrieved context contains enough evidence to answer "
                    "the question. If it does, answer using that evidence "
                    "and do not invent facts. If the context is unrelated "
                    "or insufficient, answer the user's question using "
                    "your general knowledge instead. Never claim that "
                    "general knowledge came from the document.\n\n"
                    "Retrieved context:\n" + context
                ),
            },
            {"role": "user", "content": state["question"]},
        ],
    )

    return {
        "answer": answer,
        "context": context,
        "sources": retrieved["sources"],
        "trace": retrieved["trace"] + [
            "Answered using retrieved context with general-knowledge fallback."
        ],
    }


def report_node(state: AgentState):
    """Generate a structured report based on retrieved document evidence."""
    retrieved = retrieve_context(state)
    context = retrieved["context"]

    if not context:
        return {
            "answer": (
                "Please upload and index a document before generating "
                "a document-based report."
            ),
            "sources": [],
            "context": "",
            "trace": retrieved["trace"],
        }

    report_prompt = generate_report(state["question"], context)

    answer = chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful report-writing assistant. "
                    "Use only the supplied evidence. Do not invent "
                    "facts or citations. State any limitations clearly."
                ),
            },
            {"role": "user", "content": report_prompt},
        ],
    )

    return {
        "answer": answer,
        "context": context,
        "sources": retrieved["sources"],
        "trace": retrieved["trace"]
        + ["Generated a structured report from retrieved evidence."],
    }


def general_node(state: AgentState):
    """Answer general questions using the hosted LLM."""
    answer = chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are AgentAssist. Answer clearly and accurately. "
                    "Do not claim to have used tools or documents unless "
                    "the workflow actually did so."
                ),
            },
            {"role": "user", "content": state["question"]},
        ],
    )

    return {
        "answer": answer,
        "trace": state.get("trace", [])
        + ["Answered using the general LLM response node."],
    }


def choose_next_node(state: AgentState):
    return state["route"]


builder = StateGraph(AgentState)

builder.add_node("route_question", route_question)
builder.add_node("calculator", calculator_node)
builder.add_node("document_search", document_search_node)
builder.add_node("report", report_node)
builder.add_node("general", general_node)

builder.add_edge(START, "route_question")

builder.add_conditional_edges(
    "route_question",
    choose_next_node,
    {
        "calculator": "calculator",
        "document_search": "document_search",
        "report": "report",
        "general": "general",
    },
)

builder.add_edge("calculator", END)
builder.add_edge("document_search", END)
builder.add_edge("report", END)
builder.add_edge("general", END)

agent_graph = builder.compile()


def run_agent(question, collection=None, embedding_model=None):
    """Run the AgentAssist workflow and return its execution trace."""
    result = agent_graph.invoke(
        {
            "question": question,
            "collection": collection,
            "embedding_model": embedding_model,
        }
    )

    return {
        "answer": result.get("answer", "No answer was generated."),
        "route": result.get("route", "unknown"),
        "sources": result.get("sources", []),
        "trace": result.get("trace", []),
    }
