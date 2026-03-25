# TP - IA Générative : Prompt Engineering, RAG & Agents

L'objectif de ce TP est d'expérimenter différentes techniques d'IA générative (Prompt Engineering, RAG, Agents).

Au fil des notebooks, vous allez répondre au même use-case de différentes façons : **un assistant de planification de voyage**.

---
## I. Installation

#### 1. Préparer la station de travail

Ces instructions s'appliquent uniquement dans un environnement macOS. Pour Linux ou Windows, suivez les mêmes étapes avec les commandes adaptées à votre OS.

Installer les outils système requis :

Installer git (pour cloner le projet et gérer les versions) :
```bash
brew install git
```

Installer uv (pour gérer l'environnement Python et les dépendances) :
```bash
brew install uv
uv --version
```

#### 2. Cloner le projet et changer de branche

Cloner le dépôt et se positionner dans le dossier du projet :
```bash
git clone (lien du repository)
cd TP-Formation-FDXIA
```

Changer de branche (`dev` = solutions) :
```bash
git switch dev
git pull origin dev
```

#### 3. Configurer l'environnement de travail

Créer le fichier de configuration `.env` :
```bash
cp .env.example .env
```

Renseigner dans `.env` :
- `GOOGLE_API_KEY`
- `GOOGLE_PROJECT_ID`
- `TAVILY_API_KEY`

Pour l'obtention de votre clé API Gemini et la création de votre projet merci de vous rendre sur :
https://aistudio.google.com/u/1/api-keys

De même pour Tavily :
https://www.tavily.com/

Créer l'environnement Python et installer les dépendances (dans cet ordre) :

```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

#### 4. Notebooks : choisir le bon kernel

Quand vous lancerez un notebook, votre IDE vous demandera de choisir un kernel (interpréteur Python). Sélectionnez celui de votre projet (`.venv`) pour que les dépendances soient correctement prises en compte.

Si `.venv` n'apparaît pas dans la liste des interpréteurs, assurez-vous d'avoir bien créé le dossier `.venv` et redémarrez votre IDE.

Si le notebook ne charge pas l'environnement virtuel, lancez `uv run python -c "import sys; print(sys.executable)"`, puis ajoutez ce chemin Python dans les parametres Jupyter.

---
## II. Parcours des TPs (ordre recommandé)

#### 1. Liste des notebooks à traiter

**IMPORTANT** : Par souci de clarté, vous serez amené·e à coder des fonctions dans des fichiers externes `*.py` stockés dans le dossier `shared`.

| TP  | Thème                                                   | Notebook(s)                                                    | Utilitaires dans le dossier `shared` |
|-----|---------------------------------------------------------|----------------------------------------------------------------|---------------------------------------|
| 1   | Prompt Engineering                                      | `TP1_travel_planner_LLM/1_llm_assistant.ipynb`                 | `config.py`<br>`llm_utils.py`<br>`misc_utils.py` |
| 2.1 | RAG - Préparer la base de données (version simple)      | `TP2_travel_planner_RAG/2_1_rag_db_preparation.ipynb`          | `config.py`<br>`rag_utils.py` |
| 2.2 | RAG - Assistant simple                                  | `TP2_travel_planner_RAG/2_2_rag_assistant_simple.ipynb`        | `config.py`<br>`llm_utils.py`<br>`rag_utils.py` |
| 2.3 | RAG - Préparer la base de données (version améliorée)   | `TP2_travel_planner_RAG/2_3_rag_db_improved_preparation.ipynb` | `config.py`<br>`rag_utils.py` |
| 2.4 | RAG - Assistant amélioré (Multi-Query, HyDE, reranking) | `TP2_travel_planner_RAG/2_4_rag_assistant_improved.ipynb`      | `config.py`<br>`llm_utils.py`<br>`rag_utils.py` |
| 3.1 | Agent IA outillé                                        | `TP3_travel_planner_Agent/3_1_tooling_assistant.ipynb`         | `config.py`<br>`agent_tools.py`<br>`agent_utils.py`<br>`rag_utils.py` |
| 3.2 | Agent MCP                                               | `TP3_travel_planner_Agent/3_2_mcp_assistant.ipynb`             | À compléter |



---
## III. Objectif de chaque partie

### TP1 - Prompt Engineering
- Faire un premier appel LLM
- Structurer la réponse avec un `system_prompt`
- Obtenir une sortie exploitable et industrialisable

### TP2 - RAG
- Construire une base vectorielle à partir des guides
- Récupérer les chunks pertinents pour une question utilisateur
- Générer une réponse ancrée sur les sources
- Techniques d'amélioration : Multi-Query, HyDE, Reranking

### TP3 - Agent IA
- Utiliser un agent qui choisit et enchaîne des outils
- Récupérer et combiner du contexte : date, géolocalisation, météo, recherche web et RAG
- Produire une réponse finale argumentée avec traces d'exécution


---
## IV. Comment faire ce TP ?

Lors du TP, positionnez-vous sur la branche correspondant au TP traité :
- `TP1_travel_planner_LLM`
- `TP2_travel_planner_RAG`
- `TP3_travel_planner_Agent`

```bash
git switch TPX_nom_du_TP
git pull origin TPX_nom_du_TP
```

Merci de ne pas push vos modifications afin de laisser les TP intactes.

Traitez ensuite le notebook correspondant en suivant les instructions et en complétant les parties indiquées (repérées par des `# TODO` ou `...`).

**IMPORTANT** : Chaque TP dépend des utilitaires (fichiers du dossier `shared`) développés dans les TPs précédents. *

En cas de blocage sur une fonction/classe pendant le TP, vous pouvez consulter la branche `dev` (solutions) pour débloquer et passer à la suite.

Si vous souhaitez réaliser certaines parties à votre manière, n'hésitez pas ! Suivre les questions du TP permet de faciliter la correction, mais vous êtes libre de procéder comme vous le souhaitez.

Bon Apprentissage !
