

from __future__ import annotations
from typing import Any, TYPE_CHECKING
import time
import logging

from ebr.core.base import APIEmbeddingModel
from ebr.core.meta import ModelMeta
from ebr.utils.lazy_import import LazyImport

from google import genai
from google.genai.types import EmbedContentConfig
from google.genai.errors import APIError


class GoogleEmbeddingModel(APIEmbeddingModel):
    def __init__(
        self,
        model_meta: ModelMeta,
        api_key: str | None = None,
        num_retries: int | None = None,
        **kwargs
    ):
        super().__init__(
            model_meta,
            api_key=api_key,
            num_retries=num_retries,
            **kwargs
        )
        self._client = genai.Client()

    @property
    def client(self) -> genai.Client:
        print("Initializing the client")
        if not self._client:
            self._client = genai.Client()
        return self._client

    def embed(self, data: Any, input_type: str) -> list[list[float]]:
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=data,
            config=EmbedContentConfig(
                output_dimensionality=self.embd_dim,  # Optional
            ),
        )
        return [embedding.values for embedding in response.embeddings]

    def forward(self, batch: dict[str, Any]) -> list[list[float]]:
        num_tries = 0
        while not self._num_retries or num_tries < self._num_retries:
            try:
                num_tries += 1
                result = self.embed(batch["text"], batch["input_type"][0])
            except Exception as e:
                logging.error(e)
                if isinstance(e, APIError):
                    if e.code == 429:
                        time.sleep(60)
                    elif e.code >= 500:
                        time.sleep(300)
                    else:
                        raise e
                else:
                    raise e
        return result


text_embedding_004 = ModelMeta(
    loader=GoogleEmbeddingModel,
    model_name="text-embedding-004",
    embd_dtype="float32",
    embd_dim=768,  # TODO
    num_params=1_000,  # TODO
    max_tokens=256,  # TODO
    similarity="cosine",
    reference="https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings"
)

text_embedding_005 = ModelMeta(
    loader=GoogleEmbeddingModel,
    model_name="text-embedding-005",
    embd_dtype="float32",
    embd_dim=768,  # TODO
    num_params=1_000,  # TODO
    max_tokens=256,  # TODO
    similarity="cosine",
    reference="https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings"
)
