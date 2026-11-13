# Agentic Chatbot

## Lancer l'interface

Ajoutez votre clé Google AI Studio dans `.env` (sans la publier) :

```dotenv
GEMINI_API_KEY=votre_cle_api
```

`GOOGLE_API_KEY` est également accepté. Installez les dépendances, puis
démarrez l'interface Streamlit :

```bash
uv sync
uv run streamlit run src/agentic_chatbot/app.py
```

L'application utilise le modèle Google Gemini `gemini-3.5-flash-lite`. Les réponses
sont affichées au fur et à mesure de leur génération. Les conversations enregistrées
apparaissent dans la barre latérale et peuvent être rouvertes après un redémarrage.
LangGraph `SqliteSaver` conserve leur historique dans `chatbot_state.db`. En cas de
délai dépassé, erreur de clé ou problème réseau, l'interface affiche un message
explicite.