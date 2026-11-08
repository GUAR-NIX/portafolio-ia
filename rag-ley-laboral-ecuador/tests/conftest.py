import os
import sys
from pathlib import Path

# Las pruebas nunca llaman a Ollama ni descargan el re-ranker: usan embeddings, puntajes y respuestas grabadas
os.environ["RAG_PROVEEDOR"] = "grabado"
os.environ.pop("RAG_GRABAR", None)
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ))
