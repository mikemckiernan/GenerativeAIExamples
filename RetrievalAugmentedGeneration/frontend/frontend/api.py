"""This module contains the Server that will host the frontend and API."""
import os

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from frontend.chat_client import ChatClient

from frontend import pages

STATIC_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "static")


class APIServer(FastAPI):
    """A class that hosts the service api.

    :cvar title: The title of the server.
    :type title: str
    :cvar desc: A description of the server.
    :type desc: str
    """

    title = "Chat"
    desc = "This service provides a sample conversation frontend flow."

    def __init__(self, client: ChatClient) -> None:
        """Initialize the API server."""
        self._client = client
        super().__init__(title=self.title, description=self.desc)

    def configure_routes(self) -> None:
        """Configure the routes in the API Server."""
        _ = gr.mount_gradio_app(
            self,
            blocks=pages.converse.build_page(self._client),
            path=f"/content{pages.converse.PATH}",
        )
        _ = gr.mount_gradio_app(
            self,
            blocks=pages.kb.build_page(self._client),
            path=f"/content{pages.kb.PATH}",
        )

        @self.get("/")
        async def root_redirect() -> FileResponse:
            return FileResponse(os.path.join(STATIC_DIR, "converse.html"))

        @self.get("/converse")
        async def converse_redirect() -> FileResponse:
            return FileResponse(os.path.join(STATIC_DIR, "converse.html"))

        @self.get("/kb")
        async def kb_redirect() -> FileResponse:
            return FileResponse(os.path.join(STATIC_DIR, "kb.html"))

        self.mount("/", StaticFiles(directory=STATIC_DIR, html=True))
