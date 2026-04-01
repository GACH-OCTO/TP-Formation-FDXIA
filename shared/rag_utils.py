"""Utilitaires RAG ordonnés par flux : charger -> découper -> vectoriser -> indexer -> rechercher"""

import logging
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, TypedDict

import chromadb
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from shared.config import genai_client, project_settings

# Configure logging pour les erreurs de rate limit
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


CHROMA_MAX_BATCH_SIZE = 5000


class MarkdownDocument(TypedDict):
    source: str
    text: str


class RAGChunk:
    """Chunk RAG unique utilisé partout dans le pipeline

    Champs
    - source : fichier source du chunk
    - chunk_id : identifiant local au document source
    - text : contenu texte du chunk
    - char_count : longueur du texte
    - embedding : vecteur du chunk, initialisé à [0.0] tant qu'il n'est pas calculé
    """

    def __init__(
        self,
        source: str,
        chunk_id: int | str,
        text: str,
        embedding: list[float] | None = None,
    ) -> None:
        self.source: str = source
        self.chunk_id: int | str = chunk_id
        self.text: str = text
        self.char_count: int = len(text)
        self.embedding: list[float] = embedding if embedding is not None else [0.0]


def rag_load_markdown_documents(md_dir: Path) -> list[MarkdownDocument]:
    """Charger les documents Markdown depuis un dossier

    Entrées
    - md_dir : chemin du dossier contenant les fichiers `.md`

    Sortie
    - liste `list[MarkdownDocument]` avec
      - `source` : nom du fichier
      - `text` : contenu brut du fichier
    """
    md_files = sorted(md_dir.glob("*.md"))
    documents: list[MarkdownDocument] = []

    for file_path in md_files:
        text = file_path.read_text(encoding="utf-8")
        if text.strip():
            documents.append({"source": file_path.name, "text": text})

    return documents


def rag_chunk_documents_by_chars(
    documents: list[MarkdownDocument],
    chunk_chars: int = 1200,
    chunk_overlap_chars: int = 200,
) -> list[RAGChunk]:
    """TODO découper les documents avec une fenêtre glissante par caractères (V1)

    Entrées
    - documents : liste de `MarkdownDocument`
    - chunk_chars : taille max d'un chunk
    - chunk_overlap_chars : chevauchement entre deux chunks successifs

    Sortie
    - liste de `RAGChunk`
    """
    chunks: list[RAGChunk] = []
    step = chunk_chars - chunk_overlap_chars

    for doc in tqdm(documents, desc="Découpage des documents", unit="doc"):
        chunk_id = 0
        for start_index in range(0, len(doc["text"]), step):
            chunk_text = doc["text"][start_index:start_index + chunk_chars].strip()
            if chunk_text:
                chunks.append(RAGChunk(source=doc["source"], chunk_id=chunk_id, text=chunk_text))
                chunk_id += 1

    return chunks


def rag_chunk_markdown_by_headers(
    documents: list[MarkdownDocument],
    min_chunk_chars: int = 100,
    max_chunk_chars: int = 2000,
) -> list[RAGChunk]:
    """TODO découper les documents selon la structure Markdown (V2)

    Entrées
    - documents : liste de `MarkdownDocument`
    - min_chunk_chars : taille minimale acceptée pour un chunk
    - max_chunk_chars : taille maximale avant sous-découpage

    Sortie
    - liste de `RAGChunk` avec contexte de section conservé
    """
    chunks: list[RAGChunk] = []

    for doc in documents:
        source = doc["source"]
        doc_title = source.removesuffix(".md").replace("_", " ").title()
        lines = doc["text"].splitlines()

        current_h1 = ""
        current_h2 = ""
        current_h3 = ""
        current_lines: list[str] = []
        chunk_id = 0

        def build_prefix(h1: str, h2: str, h3: str) -> str:
            prefix_lines = [f"Document: {doc_title}"]
            if h1:
                prefix_lines.append(f"Section: {h1}")
            if h2:
                prefix_lines.append(f"Sous-section: {h2}")
            if h3:
                prefix_lines.append(f"Paragraphe: {h3}")
            return "\n".join(prefix_lines)

        def flush(content_lines: list[str], h1: str, h2: str, h3: str) -> list[RAGChunk]:
            nonlocal chunk_id

            content = "\n".join(content_lines).strip()
            if not content or len(content) < min_chunk_chars:
                return []

            prefix = build_prefix(h1, h2, h3)
            full_text = prefix + "\n\n" + content
            if len(full_text) <= max_chunk_chars:
                result = [RAGChunk(source=source, chunk_id=chunk_id, text=full_text)]
                chunk_id += 1
                return result

            paragraphs = re.split(r"\n{2,}", content)
            result: list[RAGChunk] = []
            current_para_parts: list[str] = []
            current_len = 0

            for para in paragraphs:
                projected = current_len + len(para) + (2 if current_para_parts else 0)
                if current_para_parts and projected > max_chunk_chars - len(prefix) - 2:
                    para_text = prefix + "\n\n" + "\n\n".join(current_para_parts)
                    result.append(RAGChunk(source=source, chunk_id=chunk_id, text=para_text))
                    chunk_id += 1
                    current_para_parts = [para]
                    current_len = len(para)
                else:
                    current_para_parts.append(para)
                    current_len = projected

            if current_para_parts:
                para_text = prefix + "\n\n" + "\n\n".join(current_para_parts)
                result.append(RAGChunk(source=source, chunk_id=chunk_id, text=para_text))
                chunk_id += 1

            return result

        for line in lines:
            if line.startswith("#### "):
                current_lines.append(line)
            elif line.startswith("### "):
                chunks.extend(flush(current_lines, current_h1, current_h2, current_h3))
                current_h3 = line[4:].strip()
                current_lines = []
            elif line.startswith("## "):
                chunks.extend(flush(current_lines, current_h1, current_h2, current_h3))
                current_h2 = line[3:].strip()
                current_h3 = ""
                current_lines = []
            elif line.startswith("# "):
                current_h1 = line[2:].strip()
            else:
                current_lines.append(line)

        chunks.extend(flush(current_lines, current_h1, current_h2, current_h3))

    return chunks


def rag_describe_chunks(chunks: list[RAGChunk]) -> None:
    """Afficher des statistiques de taille et de répartition des chunks"""
    if not chunks:
        raise ValueError("chunks must not be empty")

    chunk_lengths = [chunk.char_count for chunk in chunks]
    source_chunk_counts = Counter(chunk.source for chunk in chunks)
    chunk_array = np.array(chunk_lengths)
    stats = {
        "count": len(chunks),
        "min length": int(chunk_array.min()),
        "median length": int(np.median(chunk_array)),
        "max length": int(chunk_array.max()),
        "mean length": int(chunk_array.mean()),
    }

    for stat_name, stat_value in stats.items():
        print(f"{stat_name}: {stat_value}")

    fig, (axis_lengths, axis_sources) = plt.subplots(1, 2, figsize=(14, 4))

    axis_lengths.hist(chunk_lengths, bins=40, color="steelblue")
    axis_lengths.set_title("Distribution des tailles de chunks")
    axis_lengths.set_xlabel("Caractères")
    axis_lengths.set_ylabel("Nombre de chunks")
    axis_lengths.set_yscale("log")

    axis_sources.bar(source_chunk_counts.keys(), source_chunk_counts.values(), color="steelblue")
    axis_sources.set_title("Chunks par source")
    axis_sources.set_xlabel("Source")
    axis_sources.set_ylabel("Nombre de chunks")
    axis_sources.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def rag_embed_texts(
    texts: list[str],
    requests_per_minute: int = 60,
    max_retries: int = 3,
) -> list[list[float]]:
    """Calculer les embeddings d'une liste de textes avec throttling
    
    Entrées
    - texts : liste des textes à vectoriser
    - requests_per_minute : nombre d'appels API autorisés par minute (default: 60)
    - max_retries : nombre de tentatives en cas d'erreur de rate limit
    
    Sortie
    - liste de vecteurs d'embeddings
    """
    # Calculer le délai minimum entre les requêtes
    min_delay_seconds = 60.0 / requests_per_minute
    
    for attempt in range(max_retries):
        try:
            response = genai_client.models.embed_content(
                model=project_settings.rag_embedding_model_name,
                contents=texts,
            )
            logger.info(f"Embeddings calculés avec succès pour {len(texts)} textes")
            # Attendre avant la prochaine requête
            time.sleep(min_delay_seconds)
            return [embedding.values for embedding in response.embeddings]
        
        except Exception as e:
            error_msg = str(e)
            is_rate_limit_error = "rate" in error_msg.lower() or "quota" in error_msg.lower() or "429" in error_msg
            
            if is_rate_limit_error:
                logger.error(f"❌ Limite de quota atteinte (tentative {attempt + 1}/{max_retries}): {error_msg}")
                
                if attempt < max_retries - 1:
                    # Augmenter le délai progressivement pour chaque retry
                    backoff_delay = min_delay_seconds * (2 ** attempt)
                    logger.info(f"⏱️ Attente de {backoff_delay:.1f}s avant nouvelle tentative...")
                    time.sleep(backoff_delay)
                else:
                    logger.error(f"❌ Impossible de calculer les embeddings après {max_retries} tentatives")
                    raise RuntimeError(
                        f"API quota limit exceeded after {max_retries} retries. "
                        f"Please reduce requests_per_minute or wait before retrying."
                    ) from e
            else:
                logger.error(f"❌ Erreur API (non-quota): {error_msg}")
                raise


def rag_build_chunk_embeddings(
    chunks: list[RAGChunk],
    batch_size: int = 16,
    requests_per_minute: int = 60,
    max_retries: int = 3,
) -> list[RAGChunk]:
    """Calculer les embeddings des chunks par batch avec contrôle de débit

    Entrées
    - chunks : liste de `RAGChunk`
    - batch_size : nombre de chunks traités par appel embedding
    - requests_per_minute : limite d'appels API par minute (default: 60)
    - max_retries : nombre de tentatives en cas de rate limit (default: 3)

    Sortie
    - liste de `RAGChunk` avec `embedding` renseigné
    """
    embedded_chunks: list[RAGChunk] = []
    
    logger.info(f"🚀 Début du calcul d'embeddings pour {len(chunks)} chunks")
    logger.info(f"⚙️ Paramètres: batch_size={batch_size}, requests_per_minute={requests_per_minute}, max_retries={max_retries}")

    for start_index in tqdm(range(0, len(chunks), batch_size), desc="Calcul des embeddings", unit="batch"):
        batch = chunks[start_index:start_index + batch_size]
        batch_texts = [f"passage: {chunk.text}" for chunk in batch]
        
        try:
            batch_vectors = rag_embed_texts(
                batch_texts,
                requests_per_minute=requests_per_minute,
                max_retries=max_retries,
            )
        except RuntimeError as e:
            logger.error(f"❌ Pipeline interrompu: {e}")
            raise

        for chunk, vector in zip(batch, batch_vectors):
            embedded_chunks.append(
                RAGChunk(
                    source=chunk.source,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    embedding=vector,
                )
            )
    
    logger.info(f"✅ Embeddings calculés avec succès pour {len(embedded_chunks)} chunks")
    return embedded_chunks


def rag_index_chunks_chroma(persist_dir: Path, chunks: list[RAGChunk]) -> None:
    """TODO indexer des chunks vectorisés dans Chroma

    Entrées
    - persist_dir : dossier de persistance Chroma
    - chunks : liste de `RAGChunk` avec embeddings calculés

    Sortie
    - None
    """
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(name="chunks")

    ids = [f"{chunk.source}_{chunk.chunk_id}" for chunk in chunks]
    docs = [chunk.text for chunk in chunks]
    embeddings = [chunk.embedding for chunk in chunks]
    metadatas = [{"source": chunk.source, "chunk_id": str(chunk.chunk_id)} for chunk in chunks]

    total = len(ids)
    for start in range(0, total, CHROMA_MAX_BATCH_SIZE):
        end = min(start + CHROMA_MAX_BATCH_SIZE, total)
        collection.add(
            ids=ids[start:end],
            documents=docs[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )


class RAGAssistant:
    """TODO assistant de recherche vectorielle sur base Chroma persistée

    Champs
    - persist_dir : dossier de base vectorielle
    - top_k : nombre de résultats par défaut
    - collection : collection Chroma `chunks`

    Méthodes
    - __init__(persist_dir, top_k) : ouverture de la collection
    - search(query, top_k) : renvoie des paires `(RAGChunk, score)`
    """

    def __init__(self, persist_dir: Path, top_k: int = 10):
        """TODO initialiser l'assistant de recherche

        Entrées
        - persist_dir : chemin de la base Chroma
        - top_k : nombre de chunks retournés par défaut
        """
        self.persist_dir: Path = persist_dir
        self.top_k: int = top_k
        if not self.persist_dir.exists():
            raise ValueError(f"persist_dir does not exist: {self.persist_dir}")

        client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection: Any = client.get_collection(name="chunks")

    def Research(self, query: str, top_k: int | None = None) -> list[tuple[RAGChunk, float]]:
        """TODO rechercher les chunks les plus proches pour une requête

        Entrées
        - query : texte utilisateur
        - top_k : surcharge optionnelle du `top_k` par défaut

        Sortie
        - liste de paires `(RAGChunk, score)`
          - `RAGChunk` : chunk retrouvé (source, chunk_id, text)
          - `score` : score de similarité dérivé de la distance vectorielle
        """
        k = top_k if top_k is not None else self.top_k
        query_embedding = rag_embed_texts([f"query: {query}"])[0]
        results = self.collection.query(query_embeddings=[query_embedding], n_results=k)

        chunks_with_scores: list[tuple[RAGChunk, float]] = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0] if results["distances"] else [0.0] * len(documents)

        for index, document_text in enumerate(documents):
            metadata = metadatas[index]
            distance = float(distances[index])
            score = 1 / (1 + distance)

            chunk_id_text = str(metadata["chunk_id"])
            chunk_id_value: int | str = int(chunk_id_text) if chunk_id_text.isdigit() else chunk_id_text
            chunk = RAGChunk(
                source=str(metadata["source"]),
                chunk_id=chunk_id_value,
                text=str(document_text),
            )
            chunks_with_scores.append((chunk, score))

        return chunks_with_scores


def rag_deduplicate_and_sort_chunks(
    chunks_with_scores: list[tuple[RAGChunk, float]],
) -> list[tuple[RAGChunk, float]]:
    """Dédupliquer les résultats par (source, chunk_id) puis trier par score décroissant"""
    best_chunks: dict[tuple[str, str], tuple[RAGChunk, float]] = {}

    for chunk, score in chunks_with_scores:
        key = (chunk.source, str(chunk.chunk_id))
        existing = best_chunks.get(key)
        if existing is None or score > existing[1]:
            best_chunks[key] = (chunk, score)

    return sorted(best_chunks.values(), key=lambda item: item[1], reverse=True)