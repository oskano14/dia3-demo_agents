import base64


class DragImage:
    ALLOWED_MIME_TYPES = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }
    MAX_IMAGES = 3
    MAX_IMAGE_SIZE = 14 * 1024 * 1024
    MAX_REQUEST_SIZE = 19 * 1024 * 1024

    def get_images(self, interaction):
        """Récupère et valide les images déposées dans st.chat_input."""
        images = list(interaction.files)

        if len(images) > self.MAX_IMAGES:
            raise ValueError(
                f"Tu peux envoyer au maximum {self.MAX_IMAGES} images à la fois."
            )

        for image in images:
            self.validate_image(image)

        return images

    def validate_image(self, image):
        """Vérifie le format, la présence et la taille d'une image."""
        if image.type not in self.ALLOWED_MIME_TYPES:
            raise ValueError(
                "Format non accepté. Utilise une image PNG, JPEG ou WebP."
            )

        image_data = image.getvalue()

        if not image_data:
            raise ValueError("L'image envoyée est vide.")

        if len(image_data) > self.MAX_IMAGE_SIZE:
            raise ValueError("Chaque image doit avoir une taille inférieure à 14 Mo.")

    def image_to_data_url(self, image):
        """Encode une image locale en URL base64 compatible avec Groq."""
        self.validate_image(image)
        image_base64 = base64.b64encode(image.getvalue()).decode("utf-8")

        return f"data:{image.type};base64,{image_base64}"

    def build_message_content(self, text, images):
        """Construit le contenu multimodal à envoyer au modèle vision."""
        if len(images) > self.MAX_IMAGES:
            raise ValueError(
                f"Tu peux envoyer au maximum {self.MAX_IMAGES} images à la fois."
            )

        content = []

        if text:
            content.append({"type": "text", "text": text})

        total_image_data_size = 0

        for image in images:
            data_url = self.image_to_data_url(image)
            total_image_data_size += len(data_url.encode("utf-8"))

            if total_image_data_size > self.MAX_REQUEST_SIZE:
                raise ValueError(
                    "La taille totale des images encodées doit rester inférieure "
                    "à 19 Mo."
                )

            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url,
                    },
                }
            )

        return content
