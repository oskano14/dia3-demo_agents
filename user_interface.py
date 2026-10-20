import streamlit as st

from llm_agent import ConversationAgent
from speech_to_text import SpeechToText


class UserInterface:
    def __init__(self):
        st.set_page_config(page_title="Jarvis")
        st.title("Barbe-Quasar")
        self.conversation_agent_object = self.get_persistent_conversation_agent()

    def get_persistent_conversation_agent(self):
        if "conversation_agent_object" not in st.session_state:
            st.session_state.conversation_agent_object = ConversationAgent()

        return st.session_state.conversation_agent_object

    def get_persistent_speech_to_text(self):
        if "speech_to_text_object" not in st.session_state:
            st.session_state.speech_to_text_object = SpeechToText()

        return st.session_state.speech_to_text_object

    def show_discussion_history(self):
        for message in self.conversation_agent_object.get_conversation_history():
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def get_interaction_text(self, interaction):
        text = interaction.text.strip()

        if interaction.audio is not None:
            with st.spinner("Transcription de l'audio avec ElevenLabs..."):
                transcription = self.get_persistent_speech_to_text().transcribe(
                    interaction.audio
                )

            if text:
                return f"{text}\n\n{transcription}"

            return transcription

        return text

    def run(self):
        self.show_discussion_history()

        interaction = st.chat_input(
            "Écris ou enregistre un message...",
            accept_file=False,
            accept_audio=True,
            audio_sample_rate=16000,
        )

        if not interaction:
            return

        try:
            text_interaction = self.get_interaction_text(interaction)
        except Exception as error:
            st.error(f"Transcription impossible : {error}")
            return

        if not text_interaction:
            st.warning("Écris un message ou enregistre ta voix avant d'envoyer.")
            return

        self.conversation_agent_object.ask_llm(text_interaction)
        st.rerun()


if __name__ == "__main__":
    user_interface_object = UserInterface()
    user_interface_object.run()
