from mcp.server.fastmcp import FastMCP


DEFAULT_TOP_K = 5
MCP_SERVER_NAME = "travel-tools-local"


mcp = FastMCP(name=MCP_SERVER_NAME)


@mcp.tool(name="rag_search")
def rag_search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, object]]:
    """Retourner les chunks RAG locaux les plus pertinents

    Entrées
    - query : requête texte
    - top_k : nombre maximal de chunks retournés

    Sortie
    - liste d'objets : source, chunk_id, text, score
    """
    from shared.agent_tools import tool_retrieve_docs

    return tool_retrieve_docs(query=query, top_k=top_k)


if __name__ == "__main__":
    mcp.run(transport="stdio")
