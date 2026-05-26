import os

from flask import Flask
from flask_cors import CORS

from app.routes import api
from app.config import CORS_ORIGINS, IS_PRODUCTION, MAX_CONTENT_LENGTH, MAX_UPLOAD_MB


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    app.config["MAX_UPLOAD_MB"] = MAX_UPLOAD_MB
    CORS(
        app,
        resources={r"/api/*": {"origins": CORS_ORIGINS}},
        supports_credentials=False,
        allow_headers=["Content-Type", "Authorization", "X-Admin-Token"],
        methods=["GET", "POST", "OPTIONS"],
        max_age=600,
    )
    app.register_blueprint(api, url_prefix="/api")
    return app


app = create_app()


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = (not IS_PRODUCTION) and os.getenv("FLASK_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
    app.run(host=host, port=port, debug=debug)
