from nodetool.nodes.openai.prediction import run_openai
import numpy as np
from nodetool.metadata.types import (
    NPArray,
)
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.chat.chat import Chunk
from nodetool.metadata.types import FunctionModel

from openai.types.create_embedding_response import CreateEmbeddingResponse
from pydantic import Field

from enum import Enum

from nodetool.workflows.types import NodeProgress


class EmbeddingModel(str, Enum):
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"


class ResponseFormat(str, Enum):
    JSON_OBJECT = "json_object"
    TEXT = "text"


class Embedding(BaseNode):
    """
    Generate vector representations of text for semantic analysis.
    embeddings, similarity, search, clustering, classification

    Uses OpenAI's embedding models to create dense vector representations of text.
    These vectors capture semantic meaning, enabling:
    - Semantic search
    - Text clustering
    - Document classification
    - Recommendation systems
    - Anomaly detection
    - Measuring text similarity and diversity
    """

    input: str = Field(title="Input", default="")
    model: EmbeddingModel = Field(
        title="Model", default=EmbeddingModel.TEXT_EMBEDDING_3_SMALL
    )
    chunk_size: int = 4096

    async def process(self, context: ProcessingContext) -> NPArray:
        # chunk the input into smaller pieces
        chunks = [
            self.input[i : i + self.chunk_size]
            for i in range(0, len(self.input), self.chunk_size)
        ]

        response = await context.run_prediction(
            self.id,
            provider="openai",
            params={"input": chunks},
            model=self.model.value,
            run_prediction_function=run_openai,
        )

        res = CreateEmbeddingResponse(**response)

        all = [i.embedding for i in res.data]
        avg = np.mean(all, axis=0)
        return NPArray.from_numpy(avg)
