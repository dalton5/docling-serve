import os
from pydantic import BaseModel, Field
import requests
import json
from collections.abc import Iterable
from typing import Any
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.models.base_model import BaseEnrichmentModel
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling_core.types.doc import (
    DoclingDocument,
    NodeItem,
    PictureItem,
)

from docling_core.types.doc.document import (
    PictureDescriptionData)

class CustomPictureDescriptionConfig(BaseModel):
    model: str = Field(default_factory=lambda: os.getenv("OPENROUTER_MODEL_PICT_DESC", "default-model"))
    prompt: str = Field(default_factory=lambda: os.getenv("OPENROUTER_MODEL_PICT_DESC_PROMPT", "Describe the image."))

class CustomPictureDescriptionPipelineOptions(PdfPipelineOptions):
    custom_picture_description: CustomPictureDescriptionConfig = Field(
        default_factory=CustomPictureDescriptionConfig
    )

class CustomPictureDescriptionEnrichmentModel(BaseEnrichmentModel):
    def __init__(self, enabled: bool, prompt: str):
        self.enabled = enabled
        self.prompt = prompt

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        return self.enabled and isinstance(element, PictureItem)

    def __call__(
        self, doc: DoclingDocument, element_batch: Iterable[NodeItem]
    ) -> Iterable[Any]:
        if not self.enabled:
            return

        for element in element_batch:
            assert isinstance(element, PictureItem)

            # uncomment this to interactively visualize the image
            # element.get_image(doc).show()

            image_url = element.image.uri.path
            description = self.call_openrouter(image_url, self.prompt)
            element.annotations.append(
                PictureDescriptionData(
                    text=description,
                    provenance='custom_picture_description',
                )
            )

            yield element


    @staticmethod
    def call_openrouter(image_url: str, prompt) -> str:
        apikey=os.getenv("OPENROUTER_API_KEY")
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {apikey}",
                "Content-Type": "application/json",
                # Optional headers:
                # "HTTP-Referer": "<YOUR_SITE_URL>",
                # "X-Title": "<YOUR_SITE_NAME>",
            },
            data=json.dumps({
                "model": os.getenv("OPENROUTER_MODEL_PICT_DESC"),
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                             {"type": "image_url", "image_url": {"url": f"data:{image_url}"}}
                        ]
                    }
                ]
            })
        )

        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter API error: {response.status_code} {response.text}")

        result = response.json()
        return result["choices"][0]["message"]["content"]

class CustomPictureDescriptionPipeline(StandardPdfPipeline):
    def __init__(self, pipeline_options: CustomPictureDescriptionPipelineOptions):
        super().__init__(pipeline_options)
        self.pipeline_options: CustomPictureDescriptionPipeline

        self.enrichment_pipe = [
            CustomPictureDescriptionEnrichmentModel(
                enabled=pipeline_options.custom_picture_description!=None, 
                prompt=pipeline_options.custom_picture_description.prompt
            )
        ]

    @classmethod
    def get_default_options(cls) -> CustomPictureDescriptionPipelineOptions:
        return CustomPictureDescriptionPipelineOptions()