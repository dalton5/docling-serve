import base64
from io import BytesIO
import os
from pathlib import Path
from urllib.parse import unquote
from pydantic import AnyUrl, BaseModel, Field
import requests
import json
from collections.abc import Iterable
from typing import Any, Optional, Union
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.models.base_model import BaseEnrichmentModel
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling_core.types.doc import (
    DoclingDocument,
    NodeItem,
    PictureItem, 
)
from docling_core.types.doc.base import (
    Size
)
from PIL import Image as PILImage

from docling_core.types.doc.document import (
    PictureDescriptionData)

class CustomPictureDescriptionConfig(BaseModel):
    model: str = Field(default_factory=lambda: os.getenv("OPENROUTER_MODEL_PICT_DESC", "default-model"))
    prompt: str = ''

class CustomPictureDescriptionPipelineOptions(PdfPipelineOptions):
    custom_picture_description: CustomPictureDescriptionConfig = Field(
        default_factory=CustomPictureDescriptionConfig
    )

class CustomPictureDescriptionEnrichmentModel(BaseEnrichmentModel):
    def __init__(self, enabled: bool, prompt: str):
        self.enabled = enabled
        self.prompt = prompt

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        return self.enabled  and isinstance(element, PictureItem)


    def __call__(
        self, doc: DoclingDocument, element_batch: Iterable[NodeItem]
    ) -> Iterable[Any]:
        if not self.enabled:
            return

        for page_no, page  in doc.pages.items():
            if page.image and page.image.uri:
                page.image.mimetype= self.call_openrouter(page.image.uri.path, self.prompt) #workarround could not stup a new property

        for element in element_batch:
            image_url, description = None, 'None'
            if isinstance(element, PictureItem):
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
        return "description"  # Placeholder for the actual description
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

        self.enrichment_pipe.append(
            CustomPictureDescriptionEnrichmentModel(
                enabled=pipeline_options.generate_page_images or pipeline_options.generate_picture_images, 
                prompt=pipeline_options.custom_picture_description.prompt
            ))
        
        print(f"CustomPictureDescriptionPipeline initialized with options: {self.enrichment_pipe}")

    @classmethod
    def get_default_options(cls) -> CustomPictureDescriptionPipelineOptions:
        return CustomPictureDescriptionPipelineOptions()