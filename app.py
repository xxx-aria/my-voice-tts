import os
import uuid
import traceback
from datetime import datetime

import gradio as gr
import torch
from TTS.api import TTS


MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# Для MVP лучше не генерировать огромные куски сразу.
# Если нужно больше — делите текст на части.
MAX_TEXT_CHARS = 1200

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

tts_model = None
model_load_error = None


def get_device() -> str:
    """
    Выбираем устройство.
    CUDA = NVIDIA GPU.
    Если CUDA нет, используем CPU.
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = get_device()


def load_model():
    """
    Ленивая загрузка модели:
    модель загружается не при старте программы, а при первой генерации.
    Так проще показать понятную ошибку в интерфейсе.
    """
    global tts_model, model_load_error

    if tts_model is not None:
        return tts_model

    try:
        print(f"Loading model: {MODEL_NAME}")
        print(f"Device: {DEVICE}")

        model = TTS(MODEL_NAME)
        model.to(DEVICE)

        tts_model = model
        model_load_error = None
        return tts_model

    except Exception as e:
        model_load_error = traceback.format_exc()
        print(model_load_error)
        raise RuntimeError(
            "Модель не загрузилась. "
            "Проверьте интернет для первого скачивания модели, установку torch/coqui-tts "
            "и наличие свободного места на диске. "
            f"Техническая ошибка: {e}"
        )


def validate_inputs(reference_audio, text: str):
    if reference_audio is None:
        raise gr.Error("Загрузите аудиофайл с вашим голосом.")

    if text is None or not text.strip():
        raise gr.Error("Введите текст для озвучки.")

    text = text.strip()

    if len(text) > MAX_TEXT_CHARS:
        raise gr.Error(
            f"Текст слишком длинный: {len(text)} символов. "
            f"Для MVP максимум {MAX_TEXT_CHARS}. "
            "Разделите текст на несколько частей."
        )

    return text


def generate_audio(reference_audio, text: str, language: str):
    """
    Генерация .wav.
    reference_audio приходит из Gradio как путь к временному файлу.
    """
    text = validate_inputs(reference_audio, text)

    try:
        model = load_model()

        file_id = uuid.uuid4().hex[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"generated_{timestamp}_{file_id}.wav")

        print("Generating audio...")
        print(f"Language: {language}")
        print(f"Reference audio: {reference_audio}")
        print(f"Output: {output_path}")

        # В разных версиях TTS параметр split_sentences может вести себя по-разному.
        # Сначала пробуем с ним, если версия API не принимает — делаем fallback.
        try:
            model.tts_to_file(
                text=text,
                speaker_wav=reference_audio,
                language=language,
                file_path=output_path,
                split_sentences=True,
            )
        except TypeError:
            model.tts_to_file(
                text=text,
                speaker_wav=reference_audio,
                language=language,
                file_path=output_path,
            )

        if not os.path.exists(output_path):
            raise RuntimeError("Файл результата не был создан.")

        return output_path, output_path

    except gr.Error:
        raise

    except Exception as e:
        details = traceback.format_exc()
        print(details)

        if model_load_error:
            raise gr.Error(
                "Модель не загрузилась. Посмотрите ошибку в терминале VS Code. "
                "Частые причины: не установлен torch, нет интернета при первом запуске, "
                "не хватает места на диске, неподходящая версия Python."
            )

        raise gr.Error(f"Ошибка генерации: {e}")


with gr.Blocks(title="Local Voice TTS MVP") as demo:
    gr.Markdown(
        """
# Local Voice TTS MVP

Загрузите короткий аудиофайл со своим голосом, введите текст и нажмите **Generate**.

⚠️ **Используй только свой голос или голос человека, который явно дал разрешение.**

Русский язык выбран по умолчанию.
        """
    )

    gr.Markdown(
        f"""
**Модель:** `{MODEL_NAME}`  
**Устройство:** `{DEVICE}`  
**Максимум текста за один раз:** `{MAX_TEXT_CHARS}` символов
        """
    )

    with gr.Row():
        with gr.Column():
            reference_audio = gr.Audio(
                label="Reference audio — ваш голос",
                sources=["upload", "microphone"],
                type="filepath",
            )

            text_input = gr.Textbox(
                label="Текст для озвучки",
                placeholder="Введите текст на русском...",
                lines=8,
                value="Привет! Это тестовая озвучка моим голосом для домашнего учебного проекта.",
            )

            language_input = gr.Dropdown(
                label="Язык",
                choices=[
                    "ru",
                    "en",
                    "es",
                    "fr",
                    "de",
                    "it",
                    "pt",
                    "pl",
                    "tr",
                    "nl",
                    "cs",
                    "ar",
                    "zh-cn",
                    "ja",
                    "hu",
                    "ko",
                    "hi",
                ],
                value="ru",
            )

            generate_button = gr.Button("Generate", variant="primary")

        with gr.Column():
            output_audio = gr.Audio(
                label="Результат",
                type="filepath",
            )

            download_file = gr.File(
                label="Скачать .wav",
            )

    generate_button.click(
        fn=generate_audio,
        inputs=[reference_audio, text_input, language_input],
        outputs=[output_audio, download_file],
    )


if __name__ == "__main__":
    print("Starting Gradio app...")
    print(f"Open local URL in browser: http://127.0.0.1:7860")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
    )