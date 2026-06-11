import json

import requests


class TelegramAPIError(RuntimeError):
    pass


class TelegramClient:
    def __init__(self, token):
        self.token = token
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.file_url = f"https://api.telegram.org/file/bot{token}"

    def api(self, method, data=None, files=None):
        response = requests.post(f"{self.base_url}/{method}", data=data, files=files, timeout=90)
        if not response.ok:
            raise TelegramAPIError(f"{method}: HTTP {response.status_code} {response.text[:300]}")
        payload = response.json()
        if not payload.get("ok"):
            raise TelegramAPIError(f"{method}: {payload}")
        return payload["result"]

    def get_updates(self, offset=None, timeout=30):
        data = {"timeout": timeout, "allowed_updates": json.dumps(["message", "callback_query"])}
        if offset is not None:
            data["offset"] = offset
        return self.api("getUpdates", data=data)

    def send_message(self, chat_id, text, reply_markup=None, parse_mode=None):
        data = {"chat_id": chat_id, "text": text[:3900]}
        if reply_markup is not None:
            data["reply_markup"] = json.dumps(reply_markup, ensure_ascii=False)
        if parse_mode is not None:
            data["parse_mode"] = parse_mode
        self.api("sendMessage", data=data)

    def send_photo(self, chat_id, path, caption=None):
        data = {"chat_id": chat_id}
        if caption:
            data["caption"] = caption[:1000]
        with path.open("rb") as image_file:
            self.api("sendPhoto", data=data, files={"photo": image_file})

    def send_document(self, chat_id, path, caption=None):
        data = {"chat_id": chat_id}
        if caption:
            data["caption"] = caption[:1000]
        with path.open("rb") as doc_file:
            self.api("sendDocument", data=data, files={"document": doc_file})

    def send_chat_action(self, chat_id, action="typing"):
        self.api("sendChatAction", data={"chat_id": chat_id, "action": action})

    def answer_callback_query(self, callback_query_id, text=None):
        data = {"callback_query_id": callback_query_id}
        if text:
            data["text"] = text
        self.api("answerCallbackQuery", data=data)

    def download_file(self, file_id, destination):
        file_info = self.api("getFile", data={"file_id": file_id})
        file_path = file_info["file_path"]
        response = requests.get(f"{self.file_url}/{file_path}", timeout=120)
        if not response.ok:
            raise TelegramAPIError(f"download: HTTP {response.status_code} {response.text[:300]}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.content)
        return destination
