# RodiumAI Bootcamp : session 4

## Lancer le serveur

Toutes les commandes se lancent depuis la racine du projet.

```bash
uv sync                     
cp .env.example .env  # Et mettre à jour les variables environement (modèle et clé api)   
uv run alembic upgrade head
uv run fastapi dev main.py
```

## Prompt système

Le prompt du tuteur Python est dans `prompts/system.md` (chargé au moment de l’appel LLM, indépendamment du répertoire de lancement). Les rôles `note` et `system-notification` restent persistés en base et visibles dans l’historique UI, mais sont exclus de l’historique envoyé au LLM.

## Streaming (`POST /chat`)

`POST /chat` répond en SSE (`text/event-stream`). Le body JSON est `{ "conversation_id", "message", "model" }`.

Événements `data:` :

- `{"type":"delta","content":"..."}` — fragment de texte (incrémental)
- `{"type":"done","reply":"...","notification":null}` — **uniquement après commit** en base
- `{"type":"error","message":"..."}` — échec ; rien n’est persisté
- `[DONE]` — fin du flux (succès ou erreur)

Tant que `done` n’est pas reçu, un texte déjà affiché au client n’est pas un message assistant enregistré. Une interruption amont ou une déconnexion client avant `done` annule le tour.

## Modèles autorisés

La liste de confiance est `ALLOWED_MODELS` dans `main.py` (cinq ids RodiumAI, ordre stable).

- `GET /models` → `{ "default": "<id>", "models": [{ "id", "label" }, ...] }`
- `POST /chat` exige `model` (id exact). Hors liste → **400** `{ "detail": "Unsupported model." }` avant tout appel LLM / stream. Champ absent → **422**.
- On peut changer de modèle d’un message à l’autre : ce n’est pas une propriété de la conversation, seulement un paramètre de chaque requête. Rien n’est stocké en base pour le modèle.
- `RODIUMAI_MODEL` (optionnel) sert uniquement de `default` pour `GET /models`. S’il est défini, il doit appartenir à la liste, sinon le serveur refuse de démarrer.

## Utiliser PostgreSQL au lieu de SQLite

Grâce à SQLAlchemy, seule la connexion change : `models.py`, `main.py` et les migrations Alembic restent identiques.

1. Installer le driver PostgreSQL :

```bash
uv add "psycopg[binary]"
```

2. `database/db.py` devient :

```python
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

load_dotenv()

# Format : postgresql+psycopg://<utilisateur>:<mot_de_passe>@<hôte>:<port>/<base>
DATABASE_URL = os.environ["DATABASE_URL"]

# pool_pre_ping : vérifie qu'une connexion est toujours ouverte avant de la réutiliser
# (un serveur PostgreSQL peut fermer les connexions inactives, contrairement à un fichier SQLite).
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(engine)


class Base(DeclarativeBase):
    pass


def get_db():
    # FastAPI dependency: one session per request, closed once the response is sent.
    with SessionLocal() as db:
        yield db
```

3. Dans `.env` :

```
DATABASE_URL=postgresql+psycopg://chatbot:chatbot@localhost:5432/chatbot
```

4. Créer les tables dans PostgreSQL (la base doit déjà exister) :

```bash
uv run alembic upgrade head
```

