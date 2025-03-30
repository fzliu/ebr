
from __future__ import annotations

import logging
import time
from abc import abstractmethod
from typing import Any, TYPE_CHECKING

from ebr.core.base import APIEmbeddingModel
from ebr.core.meta import ModelMeta
from ebr.utils.lazy_import import LazyImport

if TYPE_CHECKING:
    from google.genai.types import EmbedContentConfig
    from google.auth.transport.requests import Request
    from google.oauth2.service_account import Credentials

    from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
    from google.api_core.exceptions import ResourceExhausted, ServerError
    import vertexai
else:
    genai = LazyImport("google.genai", package_name="google-generative-ai")
    EmbedContentConfig = LazyImport("google.genai.types", "EmbedContentConfig", package_name="google-generative-ai")

    TextEmbeddingModel = LazyImport("vertexai.language_models", "TextEmbeddingModel")
    TextEmbeddingInput = LazyImport("vertexai.language_models", "TextEmbeddingInput")
    Credentials = LazyImport("google.oauth2.service_account", "Credentials")
    Request = LazyImport("google.auth.transport.requests", "Request")

    ResourceExhausted = LazyImport("google.api_core.exceptions", "ResourceExhausted")
    ServerError = LazyImport("google.api_core.exceptions", "ServerError")

    vertexai = LazyImport("vertexai")

PROJECT = "gemini-430116"
REGION = 'us-central1'
KEY_PATH = "/Users/fodizoltan/Projects/toptal/voyageai/ebr-frank/gemini-430116-c67de4bb207f.json"


class GoogleEmbeddingModel(APIEmbeddingModel):
    def __init__(
        self,
        model_meta: ModelMeta,
        num_retries: int | None = None,
        **kwargs
    ):
        super().__init__(
            model_meta,
            num_retries=num_retries,
            **kwargs
        )
        self.project = PROJECT
        self.region = REGION
        self.credentials = Credentials.from_service_account_file(
            KEY_PATH,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        if self.credentials.expired:
            self.credentials.refresh(Request())
        self._client = self.init_client()

    def forward(self, batch: dict[str, Any]) -> list[list[float]]:
        num_tries = 0
        while not self._num_retries or num_tries < self._num_retries:
            try:
                num_tries += 1
                result = self.embed(batch["text"], batch["input_type"][0])
                return result
            except Exception as e:
                self.handle_exception(e)
        raise Exception(f"Calling the API failed {num_tries} times")

    @abstractmethod
    def handle_exception(self, e: Exception):
        pass

    @abstractmethod
    def init_client(self):
        raise NotImplementedError

    @property
    def client(self) -> Any:
        return self._client

    @abstractmethod
    def embed(self, data: Any, input_type: str) -> list[list[float]]:
        raise NotImplementedError


class GoogleGenAIEmbeddingModel(GoogleEmbeddingModel):
    def handle_exception(self, e: Exception):
        logging.error(e)
        if e.code == 429:
            time.sleep(60)
        elif e.code >= 500:
            time.sleep(300)
        else:
            raise e

    def init_client(self):
        print("Initializing the genai client")
        return genai.Client(
            project=self.project,
            location=self.region,
            credentials=self.credentials,
            vertexai=True
        )

    def embed(self, data: Any, input_type: str) -> list[list[float]]:
        response = self.client.models.embed_content(
            model=self._model_meta.model_name,
            contents=data,
            config=EmbedContentConfig(
                task_type="RETRIEVAL_QUERY" if input_type == "query" else "RETRIEVAL_DOCUMENT",
                output_dimensionality=self._model_meta.embd_dim,
            ),
        )
        return [embedding.values for embedding in response.embeddings]


class GoogleVertexAIEmbeddingModel(GoogleEmbeddingModel):
    def handle_exception(self, e: Exception):
        logging.error(e)
        if isinstance(e, ResourceExhausted):  # https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/api-errors
            time.sleep(60)
        elif isinstance(e, ServerError):
            time.sleep(300)
        else:
            raise e

    def init_client(self):
        print("Initializing the vertexai client")
        vertexai.init(project=self.project, location=self.region, credentials=self.credentials)
        return TextEmbeddingModel.from_pretrained(self._model_meta.model_name)

    def embed(self, data: Any, input_type: str) -> list[list[float]]:
        task = "RETRIEVAL_QUERY" if input_type == "query" else "RETRIEVAL_DOCUMENT"
        inputs = [TextEmbeddingInput(text, task) for text in data]
        response = self.client.get_embeddings(inputs)
        return [embedding.values for embedding in response]


# genai_text_embedding_004 = ModelMeta(
#     loader=GoogleGenAIEmbeddingModel,
#     model_name="text-embedding-004",
#     embd_dtype="float32",
#     embd_dim=768,
#     max_tokens=2048,
#     similarity="cosine",
#     reference="https://ai.google.dev/gemini-api/docs/embeddings"
# )
#
# vertex_text_embedding_004 = ModelMeta(
#     loader=GoogleVertexAIEmbeddingModel,
#     model_name="text-embedding-004",
#     embd_dtype="float32",
#     embd_dim=768,
#     max_tokens=2048,
#     similarity="cosine",
#     reference="https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-text-embeddings"
# )

# genai_text_embedding_005 = ModelMeta(
#     loader=GoogleGenAIEmbeddingModel,
#     model_name="text-embedding-005",
#     embd_dtype="float32",
#     embd_dim=768,
#     max_tokens=2048,
#     similarity="cosine",
#     reference="https://ai.google.dev/gemini-api/docs/embeddings"
# )
#
# vertex_text_embedding_005 = ModelMeta(
#     loader=GoogleVertexAIEmbeddingModel,
#     model_name="text-embedding-005",
#     embd_dtype="float32",
#     embd_dim=768,
#     max_tokens=2048,
#     similarity="cosine",
#     reference="https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-text-embeddings"
# )

# genai_text_multilingual_embedding_002 = ModelMeta(
#     loader=GoogleGenAIEmbeddingModel,
#     model_name="text-multilingual-embedding-002",
#     embd_dtype="float32",
#     embd_dim=768,
#     max_tokens=2048,
#     similarity="cosine",
#     reference="https://ai.google.dev/gemini-api/docs/embeddings"
# )
#
# vertex_text_multilingual_embedding_002 = ModelMeta(
#     loader=GoogleVertexAIEmbeddingModel,
#     model_name="text-multilingual-embedding-002",
#     embd_dtype="float32",
#     embd_dim=768,
#     max_tokens=2048,
#     similarity="cosine",
#     reference="https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-text-embeddings"
# )
