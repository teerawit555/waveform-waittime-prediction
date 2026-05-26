import os

from flask import Flask
from flask_cors import CORS

from app.routes import api


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(api, url_prefix="/api")
    return app


app = create_app()


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
    app.run(host=host, port=port, debug=debug)
