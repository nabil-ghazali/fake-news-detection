import streamlit as st

from prompt.rag_system import RAGSystem

CSS = """
<style>
    /* Fixe la zone d'input en bas */
    .stChatInputContainer {
        position: fixed;
        bottom: 1rem;
        left: 0;
        right: 0;
        z-index: 100;
        background-color: white;
        padding: 1rem 2rem 0.5rem 2rem;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.05);
    }
    /* Laisse de la place en bas pour ne pas masquer le contenu */
    .block-container {
        padding-bottom: 8rem;
    }
    /* Permet le scroll si beaucoup de messages */
    .stApp {
        overflow-y: auto;
    }
</style>
"""


@st.cache_resource(show_spinner="Chargement du RAG...")
def get_rag_system() -> RAGSystem:
    """Instancie le RAG une seule fois, au lancement de l'app (jamais a l'import)."""
    return RAGSystem()


def main() -> None:
    st.set_page_config(
        page_title="Fake News Checker",
        page_icon=":newspaper:",
        layout="centered",
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    rag = get_rag_system()

    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Fake News Checker")
    st.divider()
    st.markdown(
        "Entrez une news ou un article pour vérifier la véracité de l'information."
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Écrivez ici votre news...")
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Vérification de la véracité de la news..."):
            try:
                rag.analyze_article(user_input)
                predicted_label, confidence, justification = rag.evaluation_rag()
                n_refs = len(rag.search_results["documents"][0])
                final_message = f"""
                **ANALYSIS RESULT**

                **Label :** {predicted_label}
                **Trust :** `{confidence}%`

                **Justification :**
                {justification}

                ---
                *Analysis based on comparison with {n_refs} verified articles from the database.*
                """
                st.markdown(final_message, unsafe_allow_html=True)
                st.session_state.messages.append(
                    {"role": "assistant", "content": final_message}
                )
            except Exception as e:
                error_msg = f"Erreur pendant l'analyse : {e}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )


if __name__ == "__main__":
    main()
