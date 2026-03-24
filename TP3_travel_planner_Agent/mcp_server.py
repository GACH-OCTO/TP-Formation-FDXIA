from mcp.server.fastmcp import FastMCP


DEFAULT_TOP_K = 5
MCP_SERVER_NAME = "travel-tools-local"


mcp = FastMCP(name=MCP_SERVER_NAME)

# TODO : Créez les outils locaux qui vous intéressent
# Pour l'instant, nous allons créer l'outil rag_search
# Marquez bien les types d'entrée et de sortie de vos outils, cela aidera l'agent à les utiliser correctement

@mcp.tool(name="rag_search")
def rag_search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, object]]:
    """Retourner les chunks RAG locaux les plus pertinents

    Entrées
    - query : requête texte
    - top_k : nombre maximal de chunks retournés

    Sortie
    - liste d'objets : source, chunk_id, text, score
    """
    ...
    return ...


@mcp.tool(name="...")
def my_mcp_tool_name() -> ... :
    ...
    return ...


if __name__ == "__main__":
    mcp.run(transport="stdio")
