import logging

import uvicorn

from dgxmm.app import create_app
from dgxmm.config import Settings, load_settings
from dgxmm.context import build_context


def server_kwargs(settings: Settings) -> dict:
    """Build uvicorn.run() keyword arguments from Settings."""
    return {
        "host": "0.0.0.0",
        "port": 8090,
        "proxy_headers": True,
        "forwarded_allow_ips": settings.trusted_proxies,
        "log_level": "info",
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    uvicorn.run(create_app(build_context(settings)), **server_kwargs(settings))


if __name__ == "__main__":
    main()
