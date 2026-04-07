import sys

from PySide6.QtWidgets import QApplication

from config import ALLOWED_NODES
from hardware import verify_hardware, ensure_models_exist, load_models
from network import start_subscriber
from gui import ReceiverWindow


def main():
    verify_hardware()
    ensure_models_exist()
    body_model, head_model = load_models()

    session, subscribers, camera_registry = start_subscriber()

    app = QApplication(sys.argv)
    window = ReceiverWindow(
        ALLOWED_NODES,
        body_model,
        head_model,
        session,
        subscribers,
        camera_registry,
    )
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
