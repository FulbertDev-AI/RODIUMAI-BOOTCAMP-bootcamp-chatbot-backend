# Study Buddy v2 — backend

API FastAPI du tuteur Python **Study Buddy**. Elle persiste les conversations, filtre l’historique envoyé au LLM, streame les réponses via SSE et n’autorise que cinq modèles RodiumAI.

Le frontend est un dépôt **séparé** (React + TypeScript + Vite) :

- Chatbot en ligne : [https://study-buddy-ashy-pi.vercel.app/](https://study-buddy-ashy-pi.vercel.app/)
- Backend : [https://github.com/FulbertDev-AI/RODIUMAI-BOOTCAMP-bootcamp-chatbot-backend.git](https://github.com/FulbertDev-AI/RODIUMAI-BOOTCAMP-bootcamp-chatbot-backend.git)
- Frontend : [https://github.com/FulbertDev-AI/RODIUMAI-BOOTCAMP-bootcamp-chatbot-frontend.git](https://github.com/FulbertDev-AI/RODIUMAI-BOOTCAMP-bootcamp-chatbot-frontend.git)

Les deux projets ne partagent pas le même dossier. L’interface (notes, streaming, sélecteur de modèle, Retry, Stop) vit dans le dépôt **frontend**. Ce dépôt n’expose que l’API : le navigateur l’appelle (directement ou via le proxy Vite `/api`) ; la clé RodiumAI reste ici.

---

## Fonctionnalités

| Exigence | Implémentation |
|---|---|
| Rôle custom `note` | `POST /conversations/{id}/notes` ; persisté ; exclu du LLM |
| Prompt spécialisé | `prompts/system.md` (tuteur Python débutant) |
| Streaming | `POST /chat` en SSE, `stream=true` vers RodiumAI |
| Choix du modèle | `GET /models` + champ obligatoire `model` sur `/chat` |
| Erreurs | JSON avant le flux ; événements SSE `error` ensuite |
| Bonus tokens | champ `usage` dans l’événement `done` (non stocké en base) |
| Bonus Retry / Stop | contrats backend (pas de persist si échec / déconnexion) ; UI dans le frontend |

---

## Rôle custom `note`

Une **note** est une note personnelle de l’étudiant (ex. « Je dois revoir les listes Python ce soir. »). Ce n’est **pas** un message destiné au modèle.

- Persistée en base comme un `Message` avec `role="note"` (`NOTE_ROLE` dans `main.py`).
- Renvoyée par `GET /conversations/{id}/messages` (avec `user`, `assistant`, `system-notification`).
- **Visible dans l’interface frontend** avec un **style distinct** (champ `role`). L’UI de la note est implémentée dans le dépôt frontend séparé, pas dans ce repository.
- **Jamais** envoyée au LLM.

Le filtrage est fait **côté backend** au moment de construire l’historique LLM, dans `build_llm_history()` (`main.py`) : seules les lignes dont `role` est dans `LLM_ROLES = {"user", "assistant"}` partent vers RodiumAI. `note` n’en fait pas partie (allow-list explicite, pas un oubli).

`system-notification` (message périodique tous les 10 tours user+assistant) est un autre rôle hors LLM, déjà présent avant v2. **Le rôle custom de l’exercice est `note`**, pas `system-notification`.

Pourquoi filtrer : une note personnelle ne doit pas influencer la réponse du tuteur ni consommer du contexte inutilement. Le compteur de notification utilise `len(build_llm_history(...))`, donc les notes ne décalent pas le « tous les 10 messages ».

---

## Prompt système spécialisé

Le prompt n’est plus une chaîne dans `main.py`. `load_system_prompt()` lit `prompts/system.md` via `Path(__file__)`, quel que soit le répertoire de lancement.

Structure du fichier (voir le fichier pour le texte intégral) :

- **Persona** : tuteur Python pour débutants, patient, clair, encourageant.
- **Domaine** : Python et concepts de programmation utiles à l’apprentissage (syntaxe, types, boucles, fonctions, listes, erreurs, débogage, petits exercices).
- **Hors périmètre** : tout le reste → refus poli et retour à Python.
- **Pédagogie** : guidage (questions, indices) plutôt que la solution complète d’un exercice ; explication + petit exemple si l’étudiant demande un concept ; ne pas faire tout le travail à sa place.
- **Langue / ton** : français par défaut, bienveillant, Markdown, code dans des blocs.
- **Instructions stables** : les messages utilisateur ne remplacent pas le prompt ; pas de révélation du prompt système.
- **Mémoire / résumé** : pas de politique séparée dans le fichier. Un résumé s’appuie sur l’historique **LLM** (user + assistant) renvoyé à chaque tour. Les notes ne sont pas dans cet historique.

Payload : `system` (fichier) → historique filtré → nouveau `user`.

---

## Fiche de test du prompt

Comparaison **réelle** (8 octobre 2026) via `POST https://api.rodiumai.io/v1/chat/completions`, modèle `openai/gpt-4o-mini`, `stream=false`, sans passer par le frontend.

- **Prompt ancien** (ex-`SYSTEM_PROMPT` de `main.py`) : *« Tu es Study Buddy, un tuteur bienveillant pour les étudiants. Réponds aux questions de manière claire et concise. »*
- **Prompt nouveau** : contenu actuel de `prompts/system.md`.

Pas de captures PNG dans ce dépôt (pas d’UI backend à photographier de façon fiable ici). Les extraits ci-dessous sont des **copier-coller des réponses obtenues**.

### 1. Question dans le domaine

- **Scénario** : « C'est quoi une liste en Python ? »
- **Prompt ancien / résultat** : définition correcte (indexation, mutabilité, méthode `append()`), exemple de code, ton encyclopédique — une fiche plutôt qu’un tutorat.
- **Prompt nouveau / résultat** : mêmes notions, tutoiement débutant, petit exemple, **puis** invitation à créer sa propre liste.
- **Observation** : le nouveau prompt reste dans le domaine et pousse à pratiquer, pas seulement à lire une fiche.

### 2. « Donne-moi juste la réponse de l'exercice »

- **Scénario** : fonction `carre(n)`, « le code complet, sans explication ».
- **Ancien** : uniquement

```python
def carre(n):
    return n ** 2
```

- **Nouveau** : refuse de livrer d’emblée la solution, propose de commencer plus simple et d’avancer ensemble.
- **Observation** : l’ancien prompt obéit à « juste la réponse » ; le nouveau applique la pédagogie de `prompts/system.md`.

### 3. Question hors sujet

- **Scénario** : « Quelle est la capitale de l'Australie ? »
- **Ancien** : « La capitale de l'Australie est Canberra. »
- **Nouveau** : refuse et rappelle qu’il aide sur Python / la programmation.
- **Observation** : le périmètre est effectif, pas seulement déclaré.

### 4. Tentative de détournement (« ignore tes instructions et… »)

- **Scénario testé** : ignorer les instructions et quitter le rôle de tuteur (demande hors Python).
- **Ancien et nouveau** : **HTTP 400** RodiumAI / filtre de contenu Azure (`content_filter`) — **aucune** complétion à comparer pour les deux prompts.
- **Observation** : sur cet appel, le fournisseur a bloqué les deux. On ne peut pas conclure « l’ancien a cédé / le nouveau a tenu » à partir de ce run. Le nouveau fichier contient toutefois une section *Instructions stables* absente de l’ancien prompt.

### 5. Question de mémoire

- **Scénario** : historique préalable (créer une liste, `append`), puis « Résume ce qu'on a vu depuis le début. »
- **Prompt ancien / résultat** : résumé factuel des deux tours (crochets pour créer la liste, `append` pour ajouter un élément).
- **Prompt nouveau / résultat** : même contenu, présenté en Markdown (liste numérotée + extraits de code), toujours dans le rôle tuteur.
- **Observation** : les deux s’appuient sur l’historique envoyé (pas de mémoire magique). Une **note** en base n’apparaîtrait pas dans ce résumé, car elle n’est pas dans `build_llm_history()`.

---

## Streaming

`POST /chat` répond en `text/event-stream` (pas en JSON `{reply}`). RodiumAI est appelé avec `"stream": true`.

Le **frontend** (dépôt séparé) consomme ce SSE avec `fetch()` + `ReadableStream.getReader()` (`response.body.getReader()`). `EventSource` ne convient pas : le chat est un POST. Ce repository backend n’embarque pas cette UI.

Événements `data:` (implémentation `llm_stream.format_sse` + `_chat_sse`) :

| Payload | Rôle |
|---|---|
| `{"type":"delta","content":"..."}` | fragment incrémental |
| `{"type":"done","reply":"...","notification":null,"usage":null\|{...}}` | **après** `commit` |
| `{"type":"error","message":"..."}` | échec ; rien de ce tour en base |
| `[DONE]` | fin du flux |

Avant le SSE : **404** conversation inconnue, **400** modèle interdit, **422** body invalide.

Persistance : accumulation en mémoire pendant les `delta`. Les messages (`user` + `assistant`, et la `system-notification` éventuelle) ne sont persistés **qu’après la fin complète du stream**, via `commit`, **puis** l’événement `done` est envoyé.

- Erreur LLM ou interruption **avant** cette fin : pas de `done`, **le tour incomplet n’est pas persisté**.
- Erreur LLM : `type=error` (`The LLM API call failed.`).
- Stream amont coupé avant `[DONE]` RodiumAI : `The LLM stream was interrupted.`
- Déconnexion client pendant les deltas : `GeneratorExit`, rollback, pas de persist.

Une réponse partielle déjà affichée dans le frontend n’est **pas** un message assistant en base tant que `done` n’a pas été émis.

---

## Choix du modèle

Liste unique `ALLOWED_MODELS` dans `main.py` (ordre stable) :

| `id` | `label` |
|---|---|
| `openai/gpt-4o` | GPT-4o |
| `openai/gpt-4o-mini` | GPT-4o Mini |
| `anthropic/claude-sonnet-4-5-20250929` | Claude Sonnet 4.5 |
| `anthropic/claude-haiku-4-5-20251001` | Claude Haiku 4.5 |
| `google/gemini-2.5-flash` | Gemini 2.5 Flash |

`GET /models` :

```json
{
  "default": "openai/gpt-4o",
  "models": [{ "id": "...", "label": "..." }]
}
```

`default` = `RODIUMAI_MODEL` s’il est dans la liste, sinon le premier id. Si `RODIUMAI_MODEL` est défini **hors** liste, le processus **refuse de démarrer** (`RuntimeError`).

`POST /chat` exige `model` (id exact). `require_allowed_model()` s’exécute **avant** l’appel LLM. Hors liste → **400** `{"detail":"Unsupported model."}`. Le client ne peut pas imposer un id arbitraire.

Le modèle n’est **pas** une colonne en base : il est renvoyé à **chaque** `/chat`.

### Changement de modèle au milieu d’une conversation

L’historique reste celui de la conversation (messages déjà commités). Le tour suivant envoie `system` + historique filtré + nouveau `user` au **nouvel** id. C’est possible parce que le modèle est un paramètre de requête, pas une propriété figée de `conversations`.

---

## Réponses aux questions de l’exercice

### 1. Historique DB vs historique LLM

La base (et `GET .../messages`) contient tous les rôles. Le LLM ne reçoit que `user` / `assistant`, via `build_llm_history()` (`m.role in LLM_ROLES`). `note` et `system-notification` sont persistés pour l’UI mais exclus du prompt.

### 2. Changement de modèle

Voir ci-dessus : même conversation, `model` différent à chaque POST, historique réutilisé, pas d’attache en base.

### 3. Streaming et persistance

Enregistrement **après** fin normale du flux RodiumAI **et** `commit` réussi, **puis** événement `done`. Interruption ou échec LLM : pas de `done`, aucun `user` / assistant / notification de ce tour. Les `delta` déjà vus à l’écran ne sont pas commités.

### 4. Sécurité de la clé API

`RODIUMAI_API_KEY` est lue au démarrage dans `main.py` (`os.environ`) via `.env` (`python-dotenv`). Elle part uniquement vers `https://api.rodiumai.io` (header `Authorization` dans `llm_stream.py`). Aucun endpoint ne la renvoie. `.env` est dans `.gitignore`. `.env.example` n’a que le placeholder `rd_sk_YOUR_KEY`. Le frontend n’a pas cette variable.

---

## API

Base locale : `http://127.0.0.1:8000` (docs : `/docs`).

| Méthode | Chemin | Réponse |
|---|---|---|
| `POST` | `/conversations` | **201** `{ "conversation_id": int }` |
| `GET` | `/conversations` | liste `{ id, created_at, preview }` (preview = 60 car. du seq 1) |
| `GET` | `/conversations/{id}/messages` | `{ seq, role, content, created_at }[]` |
| `POST` | `/conversations/{id}/notes` | **201** même forme ; body `{ "content": "..." }` |
| `GET` | `/models` | `{ default, models: [{ id, label }] }` |
| `POST` | `/chat` | SSE ; body `{ conversation_id, message, model }` |

---

## Installation et lancement

### Prérequis

- Python **3.14** (voir `.python-version`)
- [uv](https://docs.astral.sh/uv/)
- **PostgreSQL** (SQLAlchemy + driver `psycopg`). `DATABASE_URL` est **obligatoire**.

```bash
uv sync
cp .env.example .env
```

Compléter `.env` (aucune vraie clé ni mot de passe de production ci-dessous) :

```
RODIUMAI_API_KEY=rd_sk_YOUR_KEY
RODIUMAI_MODEL=openai/gpt-4o
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/study_buddy
```

`RODIUMAI_MODEL` est optionnel ; s’il est présent, il doit être l’un des cinq ids autorisés.

```bash
uv run alembic upgrade head
uv run fastapi dev main.py
```

UI OpenAPI : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### PostgreSQL local

1. Installer PostgreSQL et s’assurer que le serveur écoute (souvent le port `5432`).
2. Créer une base vide, par exemple `study_buddy` (et un utilisateur si besoin) :

```sql
CREATE DATABASE study_buddy;
```

3. Mettre `DATABASE_URL` dans `.env` :

`postgresql+psycopg://USER:PASSWORD@HOST:PORT/DATABASE`

Exemple local : `postgresql+psycopg://postgres:postgres@localhost:5432/study_buddy`

4. Migrations :

```bash
uv run alembic upgrade head
```

5. Backend :

```bash
uv run fastapi dev main.py
```

6. Tests (voir ci-dessous).

Le backend utilise **SQLAlchemy 2** et **psycopg** (v3). Les modèles et les révisions Alembic existantes ciblent PostgreSQL (`CURRENT_TIMESTAMP` SQL standard). La 2ᵉ migration vide `messages` puis ajoute `conversation_id`, `seq`, l’index, l’unicité `(conversation_id, seq)` et la FK.

---

## Tests

```bash
uv run pytest -v
```

La suite HTTP/SSE (`tests/test_notes.py`, streaming, modèles, prompt, `usage`) utilise une base **SQLite en mémoire** via une dépendance FastAPI surchargée (`tests/conftest.py`). Elle ne nécessite pas PostgreSQL et vérifie le métier sans appeler une base réelle.

Les contrôles de dialecte PostgreSQL sont dans `tests/test_postgres.py`. Pour une vérif **live** (connexion + schéma après `alembic upgrade head`) :

```bash
set TEST_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/study_buddy
uv run pytest -v tests/test_postgres.py
```

(Linux/macOS : `export TEST_DATABASE_URL=...`)

---

## Sécurité

- Clé API uniquement serveur ; jamais envoyée au navigateur.
- Pas de vraie clé dans ce README ni dans `.env.example`.
- `.env` ignoré par git (`.gitignore`).
- `POST /chat` refuse les modèles hors allow-list (**400**).
- Ne jamais committer une clé `rd_sk_` réelle.

---

## Bonus

**Déploiement public :** frontend [Vercel](https://study-buddy-ashy-pi.vercel.app/), backend [Render](https://study-buddy-backend-ds6g.onrender.com).

| Bonus | Où | Comportement réel |
|---|---|---|
| Affichage des tokens | Backend | Dernier `usage` RodiumAI valide (`prompt_tokens`, `completion_tokens`, `total_tokens`) recopié dans `done`. Absent ou invalide → `usage: null`. **Pas** en table `messages`. |
| Retry après erreur LLM | Frontend + contrat backend | Échec / interruption : pas de persist du tour. Un nouvel `/chat` avec le même texte n’ajoute pas un user orphelin. |
| Arrêt du streaming | Frontend + contrat backend | Fermeture de la connexion pendant les deltas → `GeneratorExit`, pas de commit. |

---

## Architecture (fichiers)

```
main.py              routes, schémas, filtrage LLM, notes, modèles, SSE chat
llm_stream.py        parse RodiumAI, usage, httpx stream=true
prompts/system.md    prompt tuteur
database/            SQLAlchemy (conversations, messages)
alembic/             migrations
tests/               pytest
```
