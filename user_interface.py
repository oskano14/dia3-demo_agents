from io import BytesIO

import streamlit as st

from drag_image import DragImage
from llm_agent import ConversationAgent
from speech_to_text import SpeechToText


class UserInterface:
    def __init__(self):
        st.set_page_config(page_title="Jarvis")
        st.title("Barbe-Quasar")
        self.conversation_agent_object = self.get_persistent_conversation_agent()
        self.image_history = self.get_persistent_image_history()
        self.drag_image_object = DragImage()

    def get_persistent_conversation_agent(self):
        if "conversation_agent_object" not in st.session_state:
            st.session_state.conversation_agent_object = ConversationAgent()
            st.session_state.image_history = {}

        return st.session_state.conversation_agent_object

    def get_persistent_image_history(self):
        if "image_history" not in st.session_state:
            st.session_state.image_history = {}

        return st.session_state.image_history

    def get_persistent_speech_to_text(self):
        if "speech_to_text_object" not in st.session_state:
            st.session_state.speech_to_text_object = SpeechToText()

        return st.session_state.speech_to_text_object

    def show_discussion_history(self):
        history = self.conversation_agent_object.get_conversation_history()

        for message_index, message in enumerate(history):
            with st.chat_message(message["role"]):
                for image in self.image_history.get(message_index, []):
                    st.image(
                        BytesIO(image["data"]),
                        caption=image["name"],
                    )

                st.markdown(message["content"])

    def remember_images(self, message_index, images):
        self.image_history[message_index] = [
            {
                "name": image.name,
                "type": image.type,
                "data": image.getvalue(),
            }
            for image in images
        ]

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
            "Écris, enregistre ou ajoute une image...",
            accept_file="multiple",
            file_type=["png", "jpg", "jpeg", "webp"],
            max_upload_size=14,
            accept_audio=True,
            audio_sample_rate=16000,
        )

        if not interaction:
            return

        try:
            text_interaction = self.get_interaction_text(interaction)
            images = self.drag_image_object.get_images(interaction)

            if images:
                message_index = len(
                    self.conversation_agent_object.get_conversation_history()
                )

                if not text_interaction:
                    text_interaction = "Décris précisément cette image."

                multimodal_content = self.drag_image_object.build_message_content(
                    text=text_interaction,
                    images=images,
                )
                self.conversation_agent_object.ask_vision_model(
                    text_interaction=text_interaction,
                    multimodal_content=multimodal_content,
                )
                self.remember_images(message_index, images)
            elif text_interaction:
                self.conversation_agent_object.ask_llm(text_interaction)
            else:
                st.warning(
                    "Écris un message, enregistre ta voix ou ajoute une image."
                )
                return
        except Exception as error:
            st.error(f"Impossible de traiter le message : {error}")
            return

        st.rerun()


if __name__ == "__main__":
    user_interface_object = UserInterface()
    user_interface_object.run()
