"""Utility modules: configuration, S3 storage, helpers."""

from src.utils.config import (
    get_llm,
    clear_llm_cache,
    check_api_connection,
    get_aws_status,
    get_active_provider_and_model,
    AVAILABLE_MODELS,
    DEFAULT_MODELS,
    GROQ_API_KEY,
    GEMINI_API_KEY,
    LLM_MODEL,
    AWS_REGION,
    S3_BUCKET_NAME,
)
from src.utils.s3_storage import (
    upload_pdf_to_s3,
    upload_text_to_s3,
    list_uploaded_documents,
    is_s3_configured,
    get_s3_console_url,
)

__all__ = [
    # config
    "get_llm",
    "check_api_connection",
    "get_aws_status",
    "GROQ_API_KEY",
    "LLM_MODEL",
    "AWS_REGION",
    "S3_BUCKET_NAME",
    # s3
    "upload_pdf_to_s3",
    "upload_text_to_s3",
    "list_uploaded_documents",
    "is_s3_configured",
    "get_s3_console_url",
]
