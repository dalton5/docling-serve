from io import BytesIO
import subprocess
import hashlib
import json
import logging
import shutil
import sys
from collections.abc import Iterable, Iterator
from functools import lru_cache
from pathlib import Path
import tempfile
import traceback
from typing import Any, Generator, Optional, Union

from fastapi import HTTPException

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.backend.pdf_backend import PdfDocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    OcrOptions,
    PdfBackend,
    PdfPipeline,
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
    PictureDescriptionVlmOptions,
    TableFormerMode,
    VlmPipelineOptions,
    smoldocling_vlm_conversion_options,
    smoldocling_vlm_mlx_conversion_options,
)
from docling.document_converter import DocumentConverter, FormatOption, PdfFormatOption
from docling.document_converter import DocumentConverter, FormatOption, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling_core.types.doc import ImageRefMode

from docling_serve.datamodel.convert import ConvertDocumentsOptions, ocr_factory
from docling_serve.helper_functions import _to_list_of_strings
from docling_serve.picturedesc_custom import CustomPictureDescriptionConfig, CustomPictureDescriptionPipeline, CustomPictureDescriptionPipelineOptions
from docling_serve.settings import docling_serve_settings

_log = logging.getLogger(__name__)


# Custom serializer for PdfFormatOption
# (model_dump_json does not work with some classes)
def _hash_pdf_format_option(pdf_format_option: PdfFormatOption) -> bytes:
    data = pdf_format_option.model_dump(serialize_as_any=True)

    # pipeline_options are not fully serialized by model_dump, dedicated pass
    if pdf_format_option.pipeline_options:
        data["pipeline_options"] = pdf_format_option.pipeline_options.model_dump(
            serialize_as_any=True, mode="json"
        )

    # Replace `pipeline_cls` with a string representation
    data["pipeline_cls"] = repr(data["pipeline_cls"])

    # Replace `backend` with a string representation
    data["backend"] = repr(data["backend"])

    # Serialize the dictionary to JSON with sorted keys to have consistent hashes
    serialized_data = json.dumps(data, sort_keys=True)
    options_hash = hashlib.sha1(
        serialized_data.encode(), usedforsecurity=False
    ).digest()
    return options_hash


# Cache of DocumentConverter objects
_options_map: dict[bytes, PdfFormatOption] = {}


@lru_cache(maxsize=docling_serve_settings.options_cache_size)
def _get_converter_from_hash(options_hash: bytes) -> DocumentConverter:
    pdf_format_option = _options_map[options_hash]
    format_options: dict[InputFormat, FormatOption] = {
        InputFormat.PDF: pdf_format_option,
        InputFormat.IMAGE: pdf_format_option,
    }

    return DocumentConverter(format_options=format_options)


def get_converter(pdf_format_option: PdfFormatOption) -> DocumentConverter:
    options_hash = _hash_pdf_format_option(pdf_format_option)
    _options_map[options_hash] = pdf_format_option
    return _get_converter_from_hash(options_hash)


def _parse_standard_pdf_opts(
    request: ConvertDocumentsOptions, artifacts_path: Optional[Path]
) -> PdfPipelineOptions:
    try:
        ocr_options: OcrOptions = ocr_factory.create_options(
            kind=request.ocr_engine.value,  # type: ignore
            force_full_page_ocr=request.force_ocr,
        )
    except ImportError as err:
        raise HTTPException(
            status_code=400,
            detail="The requested OCR engine"
            f" (ocr_engine={request.ocr_engine.value})"  # type: ignore
            " is not available on this system. Please choose another OCR engine "
            "or contact your system administrator.\n"
            f"{err}",
        )

    if request.ocr_lang is not None:
        if isinstance(request.ocr_lang, str):
            ocr_options.lang = _to_list_of_strings(request.ocr_lang)
        else:
            ocr_options.lang = request.ocr_lang

    pipeline_options = CustomPictureDescriptionPipelineOptions(
        artifacts_path=artifacts_path,
        enable_remote_services=docling_serve_settings.enable_remote_services,
        document_timeout=request.document_timeout,
        do_ocr=request.do_ocr,
        ocr_options=ocr_options,
        do_table_structure=request.do_table_structure,
        do_code_enrichment=request.do_code_enrichment,
        do_formula_enrichment=request.do_formula_enrichment,
        do_picture_classification=request.generate_detailed_pictures if request.generate_detailed_pictures else False,
        do_picture_description=request.do_picture_description,
        generate_page_images=request.generate_screenshots,
        generate_picture_images=request.generate_detailed_pictures,
        images_scale=request.images_scale if request.images_scale else 1.0,
    )
    pipeline_options.table_structure_options.mode = TableFormerMode(request.table_mode)

    # if request.image_export_mode != ImageRefMode.PLACEHOLDER:
    #     pipeline_options.generate_page_images = True
    #     if request.image_export_mode == ImageRefMode.REFERENCED :
    #         pipeline_options.generate_picture_images = True
    #     if request.image_export_mode == ImageRefMode.EMBEDDED: 
    #         pipeline_options.generate_picture_images = True
    #     if request.images_scale:
    #         pipeline_options.images_scale = request.images_scale

    pipeline_options.picture_description_options.picture_area_threshold = (
        request.picture_description_area_threshold
    )

    # if request.picture_description_local is not None:
    #     pipeline_options.picture_description_options = (
    #         PictureDescriptionVlmOptions.model_validate(
    #             request.picture_description_local.model_dump()
    #         )
    #     )

    if request.picture_description_api is not None:
        parsed_api = request.parse_picture_description_api()
        pdesc = PictureDescriptionApiOptions()
        pdesc.url = parsed_api.url
        pdesc.headers = parsed_api.headers
        pdesc.params= parsed_api.params
        pdesc.timeout = parsed_api.timeout
        pdesc.prompt = parsed_api.prompt
        pipeline_options.picture_description_options = pdesc
        pipeline_options.enable_remote_services=True

    if request.custom_picture_description is not None:
        parsed_api = request.parse_custom_picture_description()
        pipeline_options.custom_picture_description = CustomPictureDescriptionConfig()
        pipeline_options.custom_picture_description.model = parsed_api.model
        pipeline_options.custom_picture_description.prompt = parsed_api.prompt

        pipeline_options.enable_remote_services=True
    
    return pipeline_options


def _parse_backend(request: ConvertDocumentsOptions) -> type[PdfDocumentBackend]:
    if request.pdf_backend == PdfBackend.DLPARSE_V1:
        backend: type[PdfDocumentBackend] = DoclingParseDocumentBackend
    elif request.pdf_backend == PdfBackend.DLPARSE_V2:
        backend = DoclingParseV2DocumentBackend
    elif request.pdf_backend == PdfBackend.DLPARSE_V4:
        backend = DoclingParseV4DocumentBackend
    elif request.pdf_backend == PdfBackend.PYPDFIUM2:
        backend = PyPdfiumDocumentBackend
    else:
        raise RuntimeError(f"Unexpected PDF backend type {request.pdf_backend}")

    return backend


def _parse_vlm_pdf_opts(
    request: ConvertDocumentsOptions, artifacts_path: Optional[Path]
) -> VlmPipelineOptions:
    pipeline_options = VlmPipelineOptions(
        artifacts_path=artifacts_path,
        document_timeout=request.document_timeout,
    )
    pipeline_options.vlm_options = smoldocling_vlm_conversion_options
    if sys.platform == "darwin":
        try:
            import mlx_vlm  # noqa: F401

            pipeline_options.vlm_options = smoldocling_vlm_mlx_conversion_options
        except ImportError:
            _log.warning(
                "To run SmolDocling faster, please install mlx-vlm:\n"
                "pip install mlx-vlm"
            )
    return pipeline_options


# Computes the PDF pipeline options and returns the PdfFormatOption and its hash
def get_pdf_pipeline_opts(
    request: ConvertDocumentsOptions,
) -> PdfFormatOption:
    artifacts_path: Optional[Path] = None
    if docling_serve_settings.artifacts_path is not None:
        if str(docling_serve_settings.artifacts_path.absolute()) == "":
            _log.info(
                "artifacts_path is an empty path, model weights will be downloaded "
                "at runtime."
            )
            artifacts_path = None
        elif docling_serve_settings.artifacts_path.is_dir():
            _log.info(
                "artifacts_path is set to a valid directory. "
                "No model weights will be downloaded at runtime."
            )
            artifacts_path = docling_serve_settings.artifacts_path
        else:
            _log.warning(
                "artifacts_path is set to an invalid directory. "
                "The system will download the model weights at runtime."
            )
            artifacts_path = None
    else:
        _log.info(
            "artifacts_path is unset. "
            "The system will download the model weights at runtime."
        )

    pipeline_options: Union[PdfPipelineOptions, VlmPipelineOptions]
    if request.pipeline == PdfPipeline.STANDARD:
        pipeline_options = _parse_standard_pdf_opts(request, artifacts_path)
        backend = _parse_backend(request)
        pdf_format_option = PdfFormatOption(
            pipeline_cls=CustomPictureDescriptionPipeline,
            pipeline_options=pipeline_options,
            backend=backend,
        )

    elif request.pipeline == PdfPipeline.VLM:
        pipeline_options = _parse_vlm_pdf_opts(request, artifacts_path)
        pdf_format_option = PdfFormatOption(
            pipeline_cls=VlmPipeline, pipeline_options=pipeline_options
        )
    else:
        raise NotImplementedError(
            f"The pipeline {request.pipeline} is not implemented."
        )

    return pdf_format_option

SUPPORTED_EXTENSIONS = {
    ".doc", ".docx",
    ".xls", ".xlsx",
    ".ppt", ".pptx",
    ".odt", ".ods", ".odp",
    ".rtf", ".txt"
}

def convert_documents_to_pdfs(
    documents: Iterable[Union[str, Path, DocumentStream]]
) -> Generator[Union[str, Path, DocumentStream], None, None]:

    for doc in documents:
        ext = None
        input_path = None
        result_type = type(doc)
        cleanup = False

        if isinstance(doc, (str, Path)):
            input_path = Path(doc).resolve()
            ext = input_path.suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS or not input_path.exists():
                yield doc
                continue

        elif isinstance(doc, DocumentStream):
            ext = Path(doc.name).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                yield doc
                continue
            tmp_dir = tempfile.mkdtemp()
            input_path = Path(tmp_dir) / doc.name
            with open(input_path, "wb") as f:
                f.write(doc.stream.getvalue())
            cleanup = True

        else:
            yield doc
            continue

        output_path = input_path.with_suffix(".pdf")

        try:
            subprocess.run([
                "soffice", "--headless", "--convert-to", "pdf",
                "--outdir", str(input_path.parent),
                str(input_path)
            ], check=True)

            if output_path.exists():
                with open(output_path, "rb") as f:
                    pdf_content = f.read()
                    docStream = DocumentStream(stream=BytesIO(pdf_content), name=output_path.name)
                yield docStream
            else:
                yield doc

        except Exception as e:
            print(f"❌ Error converting {input_path.name if input_path else 'unknown'}: {e}")
            traceback.print_exc()
            yield doc

        finally:
            if cleanup:
                shutil.rmtree(input_path.parent, ignore_errors=True)
                
def convert_documents(
    sources: Iterable[Union[Path, str, DocumentStream]],
    options: ConvertDocumentsOptions,
    headers: Optional[dict[str, Any]] = None,
):
    
    
    sources = list(convert_documents_to_pdfs(sources))

    pdf_format_option = get_pdf_pipeline_opts(options)
    converter = get_converter(pdf_format_option)
    results: Iterator[ConversionResult] = converter.convert_all(
        sources,
        headers=headers,
        page_range=options.page_range,
        max_file_size=docling_serve_settings.max_file_size,
        max_num_pages=docling_serve_settings.max_num_pages,
    )

    return results

# def convert_to_pdf(input_path: str, output_path: str = None):
#     input_path = Path(input_path)
#     if not input_path.exists():
#         raise FileNotFoundError(f"{input_path} not found")

#     if output_path is None:
#         output_path = input_path.with_suffix(".pdf")
#     else:
#         output_path = Path(output_path)

#     ext = input_path.suffix.lower()

#     try:
#         if ext == ".docx":
#             convert_docx(str(input_path), str(output_path))

#         elif ext in [".doc", ".odt", ".ppt", ".xls", ".xlsx", ".rtf"]:
#             subprocess.run([
#                 "libreoffice", "--headless", "--convert-to", "pdf",
#                 "--outdir", str(output_path.parent),
#                 str(input_path)
#             ], check=True)

#         elif ext == ".md":
#             import markdown
#             html = markdown.markdown(input_path.read_text(encoding="utf-8"))
#             HTML(string=html).write_pdf(str(output_path))

#         elif ext in [".txt"]:
#             text = input_path.read_text(encoding="utf-8")
#             html = f"<pre>{text}</pre>"
#             HTML(string=html).write_pdf(str(output_path))

#         elif ext == ".json":
#             json_text = json.dumps(json.load(open(input_path)), indent=2)
#             html = f"<pre>{json_text}</pre>"
#             HTML(string=html).write_pdf(str(output_path))

#         elif ext == ".html":
#             HTML(filename=str(input_path)).write_pdf(str(output_path))

#         elif ext == ".epub":
#             subprocess.run([
#                 "ebook-convert", str(input_path), str(output_path)
#             ], check=True)

#         elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
#             image = Image.open(input_path).convert("RGB")
#             image.save(output_path, "PDF", resolution=100.0)

#         elif ext == ".pdf":
#             shutil.copy(str(input_path), str(output_path))

#         elif ext == ".dwg":
#             raise NotImplementedError("DWG conversion requires AutoCAD or third-party tools not available in open source.")

#         elif ext in [".fmp12", ".fp7"]:
#             raise NotImplementedError("FileMaker files must be exported as PDF manually or via FileMaker scripting.")

#         else:
#             raise ValueError(f"Unsupported file type: {ext}")

#     except Exception as e:
#         raise RuntimeError(f"Conversion failed for {input_path}: {e}")

#     return str(output_path)
