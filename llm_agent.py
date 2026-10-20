import os

from dotenv import load_dotenv
from groq import Groq


class ConversationAgent:
    TEXT_MODEL = "openai/gpt-oss-120b"
    VISION_MODEL = "qwen/qwen3.8-27b"

    def __init__(self):
        load_dotenv()
        self.groq_client = Groq(api_key=os.environ["GROQ_KEY"])
        self.conversation_history = [
            {
                "role": "system",
                "content": self.read_file(file_path="./context.txt"),
            }
        ]

    @staticmethod
    def read_file(file_path):
        with open(file_path, "r") as file:
            return file.read()

    def get_conversation_history(self):
        return self.conversation_history[1:]

    def update_conversation_history(self, role, content):
        self.conversation_history.append(
            {
                "role": role,
                "content": content,
            }
        )

    def ask_llm(self, text_interaction):
        messages = [
            *self.conversation_history,
            {"role": "user", "content": text_interaction},
        ]

        response = self.groq_client.chat.completions.create(
            messages=messages,
            model=self.TEXT_MODEL,
            include_reasoning=False,
        ).choices[0].message.content

        self.update_conversation_history(role="user", content=text_interaction)
        self.update_conversation_history(role="assistant", content=response)

        return response

    def ask_vision_model(self, text_interaction, multimodal_content):
        messages = [
            *self.conversation_history,
            {"role": "user", "content": multimodal_content},
        ]

        response = self.groq_client.chat.completions.create(
            messages=messages,
            model=self.VISION_MODEL,
        ).choices[0].message.content

        # On conserve seulement le texte, pas les lourdes données base64 des images.
        self.update_conversation_history(role="user", content=text_interaction)
        self.update_conversation_history(role="assistant", content=response)

        return response


if __name__ == "__main__":
    question = input("Pose ta question :\n")
    agent = ConversationAgent()
    response = agent.ask_llm(question)
    print("\n")
    print(response)
