import argparse
import traceback
from pathlib import Path

from .config import UPLOADS_DIR, ensure_artifact_dirs, load_settings
from .inference import PermeabilityEngine
from .preprocessing import VolumeValidationError, load_volume_file
from .presets import PRESETS, generate_preset, preset_by_key


MODEL_KEYBOARD = {
    "inline_keyboard": [
        [{"text": "Real", "callback_data": "model:real"}],
        [{"text": "Synthetic fine-tuned", "callback_data": "model:synthetic"}],
        [{"text": "Auto real/synth", "callback_data": "model:mixed"}],
    ]
}


def presets_keyboard():
    return {
        "inline_keyboard": [
            [{"text": preset.title, "callback_data": f"preset:{preset.key}"}]
            for preset in PRESETS
        ]
    }


class BotApp:
    def __init__(self, client, engine):
        self.client = client
        self.engine = engine
        self.user_models = {}
        self._preset_cache = {}

    def run_forever(self):
        offset = None
        print("Rock bot is running. Press Ctrl+C to stop.")
        while True:
            updates = self.client.get_updates(offset=offset, timeout=30)
            for update in updates:
                offset = update["update_id"] + 1
                try:
                    self.handle_update(update)
                except Exception as exc:
                    print("Update error:", exc)
                    traceback.print_exc()
                    chat_id = self._chat_id_from_update(update)
                    if chat_id is not None:
                        self.client.send_message(chat_id, f"Ошибка обработки: {exc}")

    def handle_update(self, update):
        if "callback_query" in update:
            self.handle_callback(update["callback_query"])
        elif "message" in update:
            self.handle_message(update["message"])

    def handle_callback(self, callback):
        data = callback.get("data", "")
        message = callback.get("message") or {}
        chat_id = message.get("chat", {}).get("id")
        user_id = callback.get("from", {}).get("id")
        if chat_id is None:
            return
        if data.startswith("model:"):
            model_key = data.split(":", 1)[1]
            self.user_models[self._state_key(chat_id, user_id)] = model_key
            self._answer_callback(callback, f"Выбрана модель: {model_key}")
            self.client.send_message(chat_id, f"Модель переключена на `{model_key}`. Теперь отправьте 3D-массив или выберите пресет.")
            return
        if data.startswith("preset:"):
            preset_key = data.split(":", 1)[1]
            preset = preset_by_key(preset_key)
            if preset is None:
                self._answer_callback(callback, "Пресет не найден")
                return
            self._answer_callback(callback, "Считаю пресет")
            self.client.send_message(chat_id, f"Генерирую пресет: {preset.title}")
            volume = self._preset_volume(preset.key)
            self._run_and_send(chat_id, volume, preset.key, user_id=user_id)

    def handle_message(self, message):
        chat_id = message["chat"]["id"]
        user_id = message.get("from", {}).get("id")
        text = (message.get("text") or "").strip()
        if text.startswith("/"):
            self.handle_command(chat_id, text, user_id)
            return
        if "document" in message:
            self.handle_document(chat_id, message["document"], user_id=user_id)
            return
        self.client.send_message(
            chat_id,
            "Отправьте 3D-массив документом (.npy/.npz) или используйте /presets.",
            reply_markup=presets_keyboard(),
        )

    def handle_command(self, chat_id, text, user_id=None):
        command, *args = text.split()
        command = command.split("@", 1)[0]
        if command in {"/start", "/help"}:
            self.client.send_message(chat_id, welcome_text(), reply_markup=MODEL_KEYBOARD)
            return
        if command == "/models":
            self.client.send_message(chat_id, "Выберите модель для следующего инференса:", reply_markup=MODEL_KEYBOARD)
            return
        if command == "/model":
            model_key = args[0].lower() if args else ""
            if model_key not in self.engine.specs:
                self.client.send_message(chat_id, "Доступные модели: real, synthetic, mixed.", reply_markup=MODEL_KEYBOARD)
                return
            self.user_models[self._state_key(chat_id, user_id)] = model_key
            self.client.send_message(chat_id, f"Модель переключена на `{model_key}`.")
            return
        if command == "/presets":
            self.client.send_message(chat_id, "Выберите готовый камень:", reply_markup=presets_keyboard())
            return
        self.client.send_message(chat_id, "Неизвестная команда. Попробуйте /help.")

    def handle_document(self, chat_id, document, user_id=None):
        file_name = document.get("file_name") or "volume.npy"
        size = int(document.get("file_size") or 0)
        size_mb = size / (1024 * 1024)
        if size_mb > self.engine.settings.max_upload_mb:
            self.client.send_message(chat_id, f"Файл слишком большой: {size_mb:.1f} MB. Лимит: {self.engine.settings.max_upload_mb} MB.")
            return
        suffix = Path(file_name).suffix.lower()
        if suffix not in {".npy", ".npz", ".mat", ".json", ".txt", ".csv"}:
            self.client.send_message(chat_id, "Лучше отправьте .npy или .npz с 3D-массивом. Также поддерживаются .mat/.json/.txt/.csv.")
            return
        self.client.send_message(chat_id, f"Принял `{file_name}`. Валидирую массив и считаю прогноз.")
        self.client.send_chat_action(chat_id, "upload_photo")
        destination = UPLOADS_DIR / f"{chat_id}_{document['file_id']}_{file_name}"
        path = self.client.download_file(document["file_id"], destination)
        try:
            volume = load_volume_file(path)
        except VolumeValidationError as exc:
            self.client.send_message(chat_id, f"Не удалось прочитать массив: {exc}")
            return
        self._run_and_send(chat_id, volume, Path(file_name).stem, user_id=user_id)

    def _run_and_send(self, chat_id, volume, sample_name, user_id=None):
        model_key = self.user_models.get(self._state_key(chat_id, user_id), "mixed")
        self.client.send_chat_action(chat_id, "typing")
        try:
            result = self.engine.infer(volume, model_key=model_key, sample_name=sample_name)
        except VolumeValidationError as exc:
            self.client.send_message(chat_id, f"Ошибка валидации массива: {exc}")
            return
        self.client.send_message(chat_id, result.result_text())
        for image_path in result.image_paths:
            self.client.send_chat_action(chat_id, "upload_photo")
            self.client.send_photo(chat_id, image_path)

    def _chat_id_from_update(self, update):
        if "message" in update:
            return update["message"].get("chat", {}).get("id")
        if "callback_query" in update:
            return update["callback_query"].get("message", {}).get("chat", {}).get("id")
        return None

    def _state_key(self, chat_id, user_id=None):
        return (chat_id, user_id if user_id is not None else chat_id)

    def _preset_volume(self, preset_key):
        cache_key = (preset_key, self.engine.settings.target_shape)
        if cache_key not in self._preset_cache:
            self._preset_cache[cache_key] = generate_preset(preset_key, self.engine.settings.target_shape)
        volume = self._preset_cache[cache_key]
        return volume.copy() if hasattr(volume, "copy") else volume

    def _answer_callback(self, callback, text):
        callback_id = callback.get("id")
        if not callback_id:
            return
        try:
            self.client.answer_callback_query(callback_id, text)
        except Exception as exc:
            print("Callback answer skipped:", exc)


def welcome_text():
    return (
        "Привет. Я считаю проницаемость по 3D-массиву породы.\n\n"
        "1. Выберите модель: real, synthetic или mixed (auto real/synth).\n"
        "2. Отправьте .npy/.npz документом или выберите /presets.\n"
        "3. Я верну прогноз, отчет предобработки и визуализации.\n\n"
        "Команды: /models, /presets."
    )


def run_self_test(target_shape=None, model_key="mixed"):
    settings = load_settings(target_shape_override=target_shape)
    ensure_artifact_dirs()
    engine = PermeabilityEngine(settings)
    volume = generate_preset("fractured_granite", settings.target_shape)
    result = engine.infer(volume, model_key=model_key, sample_name="self_test_fractured_granite")
    print(result.result_text())
    print("Images:")
    for path in result.image_paths:
        print(path)


def build_parser():
    parser = argparse.ArgumentParser(description="Telegram bot for rock permeability inference")
    parser.add_argument("--self-test", action="store_true", help="Run local inference without Telegram")
    parser.add_argument("--target-shape", help="Override shape, for example 64,64,64")
    parser.add_argument("--model", default="mixed", choices=["real", "synthetic", "mixed"])
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test:
        run_self_test(target_shape=args.target_shape, model_key=args.model)
        return

    settings = load_settings(target_shape_override=args.target_shape)
    ensure_artifact_dirs()
    if not settings.telegram_token:
        raise SystemExit("Не задан TELEGRAM_BOT_TOKEN. Укажите токен в .env или переменной окружения.")
    from .telegram_api import TelegramClient

    client = TelegramClient(settings.telegram_token)
    engine = PermeabilityEngine(settings)
    BotApp(client, engine).run_forever()


if __name__ == "__main__":
    main()
