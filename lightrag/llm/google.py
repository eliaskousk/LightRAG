import os
import logging
import numpy as np
from typing import Any, Union, Optional, List, Dict
from collections.abc import AsyncIterator as TypingAsyncIterator

import pipmaster as pm

# Install specific modules if not already present
if not pm.is_installed("google-genai"):
    pm.install("google-genai")
if not pm.is_installed("google-api-core"):
    pm.install("google-api-core")
if not pm.is_installed("numpy"):
    pm.install("numpy")  # numpy is used for embeddings
if not pm.is_installed("tenacity"):
    pm.install("tenacity")  # tenacity for retries
if not pm.is_installed("pydantic"):  # For response_schema if using Pydantic models
    pm.install("pydantic")

from google.genai import Client as GoogleGenAIClient
from google.genai import types as google_types
from google.api_core import exceptions as google_api_exceptions

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from lightrag.types import GPTKeywordExtractionFormat

from lightrag.utils import (
    wrap_embedding_func_with_attrs,
    # locate_json_string_body_from_string, # May not be needed if response_schema is robust
    safe_unicode_decode,
    remove_think_tags,
    logger,
    verbose_debug,
    VERBOSE_DEBUG,
)
from lightrag.api import __api_version__  # For User-Agent

# Load environment variables from.env file if present
# This allows local.env to override global env vars if override=True,
# but LightRAG's typical pattern is override=False (OS takes precedence)
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)


class InvalidResponseError(Exception):
    """Custom exception class for triggering retry mechanism on invalid/empty responses."""

    pass


# Define retryable exceptions for Google API calls
RETRYABLE_GOOGLE_EXCEPTIONS = (
    google_api_exceptions.ResourceExhausted,  # HTTP 429
    google_api_exceptions.ServiceUnavailable,  # HTTP 503
    google_api_exceptions.DeadlineExceeded,  # HTTP 504
    google_api_exceptions.InternalServerError,  # HTTP 500
    google_api_exceptions.Unknown,  # HTTP 500 (often)
    google_api_exceptions.Aborted,  # Context-dependent, can be retryable
    google_api_exceptions.GatewayTimeout,  # HTTP 504
    google_api_exceptions.BadGateway,  # HTTP 502
    InvalidResponseError,  # Custom error for empty/malformed success
)


DEFAULT_GOOGLE_GEMINI_MODEL = "gemini-3-flash-preview"
DEFAULT_GOOGLE_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_GOOGLE_EMBEDDING_DIM = 768  # Using MRL to reduce from default 3072
DEFAULT_GOOGLE_MAX_TOKEN_SIZE = 8192

# Global client cache
_client_cache: Dict[str, GoogleGenAIClient] = {}


def clear_client_cache() -> None:
    """
    Clear the global client cache.
    Useful for testing or when configuration changes.
    """
    global _client_cache
    _client_cache.clear()
    logger.info("Google GenAI client cache cleared")


def _get_client_cache_key(
    api_key: Optional[str] = None,
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    use_vertex_ai: Optional[bool] = None,
) -> str:
    """
    Generate a cache key for the client based on connection parameters.
    """
    # Determine effective values
    effective_api_key = (
        api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    )
    effective_project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
    effective_location = location or os.environ.get("GOOGLE_CLOUD_LOCATION")
    
    # Determine use_vertex_ai
    effective_use_vertex_ai = use_vertex_ai
    if effective_use_vertex_ai is None:
        if effective_project_id:
            effective_use_vertex_ai = True
        elif effective_api_key:
            effective_use_vertex_ai = False
        else:
            effective_use_vertex_ai = False
    
    # Create cache key
    if effective_use_vertex_ai:
        return f"vertex_{effective_project_id}_{effective_location}"
    else:
        # Hash the API key for security (don't store plain API keys as cache keys)
        import hashlib
        key_hash = hashlib.sha256(effective_api_key.encode()).hexdigest()[:16] if effective_api_key else "no_key"
        return f"gemini_{key_hash}"


async def create_google_async_client(
    api_key: Optional[str] = None,
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    use_vertex_ai: Optional[bool] = None,
    client_configs: Optional[Dict[str, Any]] = None,
    use_cache: bool = True,
    base_url: Optional[str] = None,
    timeout: Optional[int] = None,
) -> GoogleGenAIClient:
    """
    Creates an asynchronous Google Generative AI client.
    Prioritizes explicit params, then environment variables.
    Uses a cache to avoid recreating clients with the same configuration.
    """
    # Check cache first if enabled
    if use_cache:
        cache_key = _get_client_cache_key(api_key, project_id, location, use_vertex_ai)
        if cache_key in _client_cache:
            logger.debug(f"Using cached Google GenAI client (cache_key: {cache_key})")
            return _client_cache[cache_key]
    effective_use_vertex_ai = use_vertex_ai

    # Determine API key
    effective_api_key = (
        api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    )

    # Determine Vertex AI params
    effective_project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
    effective_location = location or os.environ.get("GOOGLE_CLOUD_LOCATION")

    # Infer use_vertex_ai if not explicitly set by env var parsing
    if effective_use_vertex_ai is None:
        if effective_project_id:  # If project_id is available, assume Vertex AI unless API key is also present and dominant
            effective_use_vertex_ai = True
        elif effective_api_key:  # If only API key is available, assume Gemini API
            effective_use_vertex_ai = False
        else:  # Default to Gemini API if no clear indicators, will likely fail if no API key
            effective_use_vertex_ai = False
            logger.warning(
                "Could not determine Google API mode (Vertex AI vs Gemini API key). Defaulting to Gemini API. Ensure GEMINI_API_KEY is set."
            )

    # User-Agent header
    default_headers = {
        "User-Agent": f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_8) LightRAG/{__api_version__}",
        # Content-Type is typically handled by the SDK per request
    }
    http_options: Dict[str, Any] = {"headers": default_headers}

    # Add base_url (api_endpoint) if provided
    if base_url:
        http_options["api_endpoint"] = base_url

    # Add timeout if provided (convert seconds to milliseconds for Gemini API)
    if timeout is not None:
        http_options["timeout"] = timeout * 1000

    if client_configs and "http_options" in client_configs:
        http_options.update(client_configs.pop("http_options"))

    merged_client_args = client_configs.copy() if client_configs else {}
    merged_client_args["http_options"] = google_types.HttpOptions(**http_options)

    if effective_use_vertex_ai:
        logger.info("Initializing Google GenAI Client for Vertex AI.")
        if not effective_project_id:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT environment variable or project_id parameter must be set for Vertex AI."
            )
        # Location is recommended, some models might require it.
        if not effective_location:
            logger.warning(
                "GOOGLE_CLOUD_LOCATION or location parameter not set for Vertex AI. This might lead to errors or default region usage."
            )

        final_vertex_args = {
            "vertexai": True,
            "project": effective_project_id,
            "location": effective_location,
        }

        final_vertex_args.update(merged_client_args)
        client = GoogleGenAIClient(**final_vertex_args)
        
        # Cache the client if caching is enabled
        if use_cache:
            cache_key = _get_client_cache_key(api_key, project_id, location, use_vertex_ai)
            _client_cache[cache_key] = client
            logger.debug(f"Cached new Vertex AI client (cache_key: {cache_key})")
        
        return client
    else:
        logger.info("Initializing Google GenAI Client for Gemini API (API Key).")
        if not effective_api_key:
            raise ValueError(
                "GEMINI_API_KEY/GOOGLE_API_KEY environment variable or api_key parameter must be set for Gemini API."
            )

        final_gemini_args = {"api_key": effective_api_key}
        final_gemini_args.update(merged_client_args)
        client = GoogleGenAIClient(**final_gemini_args)
        
        # Cache the client if caching is enabled
        if use_cache:
            cache_key = _get_client_cache_key(api_key, project_id, location, use_vertex_ai)
            _client_cache[cache_key] = client
            logger.debug(f"Cached new Gemini API client (cache_key: {cache_key})")
        
        return client


def _extract_response_text(
    response: Any, extract_thoughts: bool = False
) -> tuple[str, str]:
    """
    Extract text content from Gemini response, separating regular content from thoughts.

    Args:
        response: Gemini API response object
        extract_thoughts: Whether to extract thought content separately

    Returns:
        Tuple of (regular_text, thought_text)
    """
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return ("", "")

    regular_parts: List[str] = []
    thought_parts: List[str] = []

    for candidate in candidates:
        if not getattr(candidate, "content", None):
            continue
        # Use 'or []' to handle None values from parts attribute
        for part in getattr(candidate.content, "parts", None) or []:
            text = getattr(part, "text", None)
            if not text:
                continue

            # Check if this part is thought content using the 'thought' attribute
            is_thought = getattr(part, "thought", False)

            if is_thought and extract_thoughts:
                thought_parts.append(text)
            elif not is_thought:
                regular_parts.append(text)

    return ("\n".join(regular_parts), "\n".join(thought_parts))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(RETRYABLE_GOOGLE_EXCEPTIONS),
    reraise=True,
)
async def google_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    use_vertex_ai: Optional[bool] = None,
    token_tracker: Optional[Any] = None,
    enable_cot: bool = False,
    base_url: Optional[str] = None,
    timeout: Optional[int] = None,
    **kwargs: Any,
) -> Union[str, TypingAsyncIterator[str]]:
    """
    Core function to complete a prompt using Google's Generative AI API.
    Handles client creation, request formatting, API call, and response processing.

    This function supports automatic integration of reasoning content from Gemini models
    that provide Chain of Thought capabilities via the thinking_config API feature.

    COT Integration:
    - When enable_cot=True: Thought content is wrapped in <think>...</think> tags
    - When enable_cot=False: Thought content is filtered out, only regular content returned
    - Thought content is identified by the 'thought' attribute on response parts

    Args:
        model: The Gemini model to use.
        prompt: The prompt to complete.
        system_prompt: Optional system prompt to include.
        history_messages: Optional list of previous messages in the conversation.
        api_key: Optional API key. If None, uses environment variable.
        project_id: Optional Google Cloud project ID for Vertex AI.
        location: Optional Google Cloud location for Vertex AI.
        use_vertex_ai: Whether to use Vertex AI instead of Gemini API.
        token_tracker: Optional token usage tracker for monitoring API usage.
        enable_cot: Whether to include Chain of Thought content in the response.
        base_url: Optional custom API endpoint.
        timeout: Request timeout in seconds (converted to milliseconds for Gemini API).
        **kwargs: Additional generation parameters (temperature, max_output_tokens, etc.)

    Returns:
        The completed text (with COT content if enable_cot=True) or an async iterator
        of text chunks if streaming. COT content is wrapped in <think>...</think> tags.

    Raises:
        InvalidResponseError: If the response from Google API is empty or blocked.
        ValueError: If API key or Vertex AI configuration is not provided.
    """
    if not VERBOSE_DEBUG and logging.getLogger("google_genai").level != logging.WARNING:
        logging.getLogger("google_genai").setLevel(
            logging.WARNING
        )  # Reduce verbosity of underlying SDK

    client_call_configs = kwargs.pop("google_client_configs", {})  # For client creation
    use_cache = kwargs.pop("use_client_cache", True)  # Allow disabling cache if needed

    google_client = await create_google_async_client(
        api_key=api_key,
        project_id=project_id,
        location=location,
        use_vertex_ai=use_vertex_ai,
        client_configs=client_call_configs,
        use_cache=use_cache,
        base_url=base_url,
        timeout=timeout,
    )

    # Prepare contents for API call
    api_contents: List[google_types.Content] = []
    if history_messages:
        for msg in history_messages:
            role = msg.get("role", "user").lower()
            # Google SDK expects "user" or "model"
            if role not in ["user", "model"]:
                logger.debug(
                    f"Invalid role '{role}' in history_messages, mapping to 'user'. Supported: 'user', 'model'."
                )
                role = "user" if role != "assistant" else "model"  # common mapping
            api_contents.append(
                google_types.Content(
                    role=role,
                    parts=[google_types.Part(text=str(msg.get("content", "")))],
                )
            )

    api_contents.append(
        google_types.Content(role="user", parts=[google_types.Part(text=prompt)])
    )

    # Prepare GenerateContentConfig
    gen_config_params = {}
    if system_prompt:
        # For google-genai, system_instruction is part of GenerateContentConfig or GenerativeModel constructor
        # Here, we pass it via GenerateContentConfig for per-call flexibility.
        gen_config_params["system_instruction"] = google_types.Content(
            role="system", parts=[google_types.Part(text=system_prompt)]
        )

    # Standard generation parameters from kwargs
    for param_name in [
        "temperature",
        "max_output_tokens",
        "top_p",
        "top_k",
        "candidate_count",
        "seed",
        "stop_sequences",
        "presence_penalty",
        "frequency_penalty",
    ]:
        if param_name in kwargs:
            gen_config_params[param_name] = kwargs[param_name]
    if (
        "stop_sequences" in kwargs and kwargs["stop_sequences"]
    ):  # Ensure it's not None or empty
        gen_config_params["stop_sequences"] = kwargs["stop_sequences"]

    # JSON mode parameters (response_mime_type and response_schema)
    response_mime_type = kwargs.get("response_mime_type")
    response_schema = kwargs.get("response_schema")

    if response_mime_type:
        gen_config_params["response_mime_type"] = response_mime_type
    if response_schema:
        gen_config_params["response_schema"] = response_schema
        if not response_mime_type:  # Default to application/json if schema is provided
            gen_config_params["response_mime_type"] = "application/json"
            logger.debug(
                "response_schema provided without response_mime_type, defaulting to application/json."
            )

    # Safety settings (example, can be made configurable via kwargs)
    safety_settings_obj = kwargs.get("safety_settings")
    if safety_settings_obj:
        gen_config_params["safety_settings"] = safety_settings_obj

    generation_config_obj = (
        google_types.GenerateContentConfig(**gen_config_params)
        if gen_config_params
        else None
    )

    logger.debug("===== Entering func of Google LLM =====")
    logger.debug(f"Model: {model}")
    logger.debug(f"GenerateContentConfig effective params: {gen_config_params}")
    logger.debug(
        f"Num of history messages (converted to Content objects): {len(api_contents) - 1}"
    )
    verbose_debug(f"System prompt (via GenerateContentConfig): {system_prompt}")
    verbose_debug(f"User Query (latest): {prompt}")
    logger.debug("===== Sending Query to Google LLM =====")

    is_streaming = kwargs.get("stream", False)

    try:
        if is_streaming:
            response_iter = await google_client.aio.models.generate_content_stream(
                model=model,
                contents=api_contents,
                config=generation_config_obj,
            )

            async def stream_generator():
                full_response_text_for_log = []
                # COT state tracking for streaming
                cot_active = False
                cot_started = False
                initial_content_seen = False

                try:
                    async for chunk in response_iter:
                        # Extract both regular and thought content from chunk
                        regular_text, thought_text = _extract_response_text(
                            chunk, extract_thoughts=True
                        )

                        # Handle empty chunks
                        if not regular_text and not thought_text:
                            # Check for finish_reason metadata
                            if (
                                hasattr(chunk, "candidates")
                                and chunk.candidates
                            ):
                                for candidate in chunk.candidates:
                                    if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                                        logger.debug(
                                            f"Stream chunk finish_reason: {candidate.finish_reason.name}"
                                        )
                            continue

                        if enable_cot:
                            # Process regular content
                            if regular_text:
                                if not initial_content_seen:
                                    initial_content_seen = True

                                # Close COT section if it was active
                                if cot_active:
                                    yield "</think>"
                                    cot_active = False

                                # Send regular content
                                content_text = regular_text
                                if r"\u" in content_text:
                                    content_text = safe_unicode_decode(
                                        content_text.encode("utf-8")
                                    )
                                full_response_text_for_log.append(content_text)
                                yield content_text

                            # Process thought content
                            if thought_text:
                                if not initial_content_seen and not cot_started:
                                    # Start COT section
                                    yield "<think>"
                                    cot_active = True
                                    cot_started = True

                                # Send thought content if COT is active
                                if cot_active:
                                    content_text = thought_text
                                    if r"\u" in content_text:
                                        content_text = safe_unicode_decode(
                                            content_text.encode("utf-8")
                                        )
                                    full_response_text_for_log.append(content_text)
                                    yield content_text
                        else:
                            # COT disabled - only send regular content
                            if regular_text:
                                content_text = regular_text
                                if r"\u" in content_text:
                                    content_text = safe_unicode_decode(
                                        content_text.encode("utf-8")
                                    )
                                full_response_text_for_log.append(content_text)
                                yield content_text

                    # Ensure COT is properly closed if still active
                    if cot_active:
                        yield "</think>"

                except Exception as e:
                    # Try to close COT tag before reporting error
                    if cot_active:
                        try:
                            yield "</think>"
                        except Exception:
                            pass
                    logger.error(f"Error during Google API stream processing: {e}")
                    raise
                finally:
                    logger.debug(
                        f"Stream ended. Full streamed response length: {len(''.join(full_response_text_for_log))}"
                    )
                    verbose_debug(
                        f"Full streamed response: {''.join(full_response_text_for_log)}"
                    )

            return stream_generator()
        else:  # Non-streaming
            response = await google_client.aio.models.generate_content(
                model=model,
                contents=api_contents,
                config=generation_config_obj,
            )

            # Extract both regular text and thought text using the helper
            regular_text, thought_text = _extract_response_text(response, extract_thoughts=True)

            # Check for empty response and blocking reasons
            if not regular_text and not thought_text:
                # Check for blocking reasons
                if (
                    response
                    and hasattr(response, "prompt_feedback")
                    and response.prompt_feedback
                    and response.prompt_feedback.block_reason
                ):
                    err_msg = f"Google API request was blocked. Reason: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason.name}"
                    logger.error(err_msg)
                    raise InvalidResponseError(err_msg)

                # Check candidates for blocking
                if response and hasattr(response, "candidates") and response.candidates:
                    for candidate in response.candidates:
                        if (
                            hasattr(candidate, "finish_reason")
                            and candidate.finish_reason
                            and candidate.finish_reason.name == "SAFETY"
                        ):
                            err_msg = f"Google API response candidate blocked due to safety. Ratings: {candidate.safety_ratings}"
                            logger.error(err_msg)
                            raise InvalidResponseError(err_msg)

                logger.error(
                    "Received empty or invalid content from Google API non-streaming response."
                )
                raise InvalidResponseError("Received empty content from Google API.")

            # Apply COT filtering logic based on enable_cot parameter
            if enable_cot:
                # Include thought content wrapped in <think> tags
                if thought_text and thought_text.strip():
                    if not regular_text or regular_text.strip() == "":
                        # Only thought content available
                        content_to_return = f"<think>{thought_text}</think>"
                    else:
                        # Both content types present: prepend thought to regular content
                        content_to_return = f"<think>{thought_text}</think>{regular_text}"
                else:
                    # No thought content, use regular content only
                    content_to_return = regular_text or ""
            else:
                # Filter out thought content, return only regular content
                content_to_return = regular_text or ""

            if not content_to_return:
                raise InvalidResponseError("Google API response did not contain any text content.")

            if r"\u" in content_to_return:
                content_to_return = safe_unicode_decode(
                    content_to_return.encode("utf-8")
                )

            # Remove think tags if not in COT mode (for consistency)
            content_to_return = remove_think_tags(content_to_return)

            # Token tracking for non-streaming
            if (
                token_tracker
                and hasattr(response, "usage_metadata")
                and response.usage_metadata
            ):
                usage = response.usage_metadata
                token_counts = {
                    "prompt_tokens": getattr(usage, "prompt_token_count", 0),
                    "completion_tokens": getattr(usage, "candidates_token_count", 0),
                    "total_tokens": getattr(usage, "total_token_count", 0)
                    or (
                        getattr(usage, "prompt_token_count", 0)
                        + getattr(usage, "candidates_token_count", 0)
                    ),
                }
                token_tracker.add_usage(token_counts)
                logger.debug(f"Google API token usage: {token_counts}")

            logger.debug(
                f"Google API Response content length: {len(content_to_return)}"
            )
            verbose_debug(
                f"Google API Response: {content_to_return[:500]}{'...' if len(content_to_return) > 500 else ''}"
            )

            return content_to_return

    except google_api_exceptions.GoogleAPIError as e:
        logger.error(f"Google API Error: {e.__class__.__name__} - {e}")
        # Specific handling for common non-retryable errors if needed, though tenacity handles retryable ones
        if isinstance(
            e, google_api_exceptions.InvalidArgument
        ):  # Typically non-retryable
            logger.error(
                f"Google API Invalid Argument: {e}. This is often due to malformed request or invalid model parameters."
            )
        elif isinstance(e, google_api_exceptions.PermissionDenied):
            logger.error(
                f"Google API Permission Denied: {e}. Check credentials and API enablement."
            )
        raise  # Reraise for tenacity or higher-level handling
    except Exception as e:
        logger.error(
            f"Unexpected error during Google API call: {e.__class__.__name__} - {e}"
        )
        raise


async def google_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[dict]] = None,
    keyword_extraction: bool = False,
    **kwargs: Any,
) -> Union[str, TypingAsyncIterator[str]]:
    """
    Simplified wrapper for Google text completion.
    Determines model name and sets up for keyword extraction if requested.
    """
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]

    if "gemini" not in model_name:
        # TODO check against client.models.list()
        logger.warning(
            f"Invalid `llm_model_name` Argument: {model_name}. Set a correct model name - default to {DEFAULT_GOOGLE_GEMINI_MODEL}."
        )
        # Fallback to environment variable or a hardcoded default
        model_name = os.environ.get("LLM_MODEL", DEFAULT_GOOGLE_GEMINI_MODEL)

    # Keyword extraction setup
    keyword_extraction = kwargs.pop("keyword_extraction", keyword_extraction)
    if keyword_extraction:
        kwargs["response_mime_type"] = "application/json"
        # Use the Pydantic model defined earlier or a dict schema
        kwargs["response_schema"] = GPTKeywordExtractionFormat
        logger.debug(
            "Keyword extraction enabled, setting response_mime_type to application/json and providing schema."
        )

    # API key and Vertex params can be passed via kwargs or picked up from env by create_google_async_client
    api_key = kwargs.pop("api_key", None)
    project_id = kwargs.pop("project_id", None)
    location = kwargs.pop("location", None)
    use_vertex_ai = kwargs.pop("use_vertex_ai", None)

    if use_vertex_ai is None:
        # Determine if GOOGLE_GENAI_USE_VERTEXAI was true from environment
        use_vertex_ai_str = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "false").lower()
        use_vertex_ai = use_vertex_ai_str == "true"

    result = await google_complete_if_cache(
        model=model_name,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=api_key,
        project_id=project_id,
        location=location,
        use_vertex_ai=use_vertex_ai,
        **kwargs,  # Remaining kwargs (token_tracker, temperature, etc.)
    )

    if (
        keyword_extraction
        and isinstance(result, str)
        and kwargs.get("response_mime_type") == "application/json"
    ):
        # If the model still wraps the JSON in text, try to extract it.
        # However, with response_schema, this should ideally not be needed.
        # The locate_json_string_body_from_string is a fallback.
        # json_body = locate_json_string_body_from_string(result)
        # return json_body if json_body else result
        return result  # Assuming response_schema ensures result is a clean JSON string

    return result


# --- Specific Model Wrappers (Examples) ---
async def gemini_2_5_flash_lite_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[dict]] = None,
    keyword_extraction: bool = False,
    **kwargs: Any,
) -> Union[str, dict]:
    # Keyword extraction setup
    keyword_extraction = kwargs.pop("keyword_extraction", keyword_extraction)
    if keyword_extraction:
        kwargs["response_mime_type"] = "application/json"
        # Use the Pydantic model defined earlier or a dict schema
        kwargs["response_schema"] = GPTKeywordExtractionFormat
        logger.debug(
            "Keyword extraction enabled, setting response_mime_type to application/json and providing schema."
        )
    return await google_complete_if_cache(
        "gemini-2.5-flash-lite",
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(
    embedding_dim=DEFAULT_GOOGLE_EMBEDDING_DIM,
    max_token_size=DEFAULT_GOOGLE_MAX_TOKEN_SIZE,  # Max tokens for a single text input
)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(
        multiplier=1, min=4, max=60
    ),  # Longer max wait for embeddings
    retry=retry_if_exception_type(RETRYABLE_GOOGLE_EXCEPTIONS),
    reraise=True,
)
async def google_embed(
    texts: List[str],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    use_vertex_ai: Optional[bool] = None,
    client_configs: Optional[dict] = None,
    task_type: Optional[str] = None,
    title: Optional[str] = None,
    output_dimensionality: Optional[int] = None,
    base_url: Optional[str] = None,
    timeout: Optional[int] = None,
    **kwargs: Any,
) -> np.ndarray:
    """
    Generates embeddings for a list of texts using Google's embedding models.

    This function uses Google's Gemini embedding model to generate text embeddings.
    It supports dynamic dimension control and automatic L2 normalization for dimensions
    less than 3072.

    Args:
        texts: List of texts to embed.
        model: The Gemini embedding model to use. Default is from env or "gemini-embedding-001".
        api_key: Optional API key. If None, uses environment variables.
        project_id: Optional Google Cloud project ID for Vertex AI.
        location: Optional Google Cloud location for Vertex AI.
        use_vertex_ai: Whether to use Vertex AI instead of Gemini API.
        client_configs: Optional additional client configuration.
        task_type: Task type for embedding optimization (e.g., "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY").
        title: Optional title for RETRIEVAL_DOCUMENT task_type.
        output_dimensionality: Optional embedding dimension for dynamic dimension reduction.
            Supported range: 128-3072. Recommended values: 768, 1536, 3072.
        base_url: Optional custom API endpoint.
        timeout: Request timeout in seconds (converted to milliseconds for Gemini API).
        **kwargs: Additional parameters.

    Returns:
        A numpy array of embeddings, one per input text. For dimensions < 3072,
        the embeddings are L2-normalized to ensure optimal semantic similarity performance.

    Raises:
        InvalidResponseError: If the response from Google API is invalid or empty.
    """
    if not texts:
        return np.array()

    if not VERBOSE_DEBUG and logging.getLogger("google_genai").level != logging.WARNING:
        logging.getLogger("google_genai").setLevel(logging.WARNING)

    if use_vertex_ai is None:
        # Determine if GOOGLE_GENAI_USE_VERTEXAI was true from environment
        use_vertex_ai_str = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "false").lower()
        use_vertex_ai = use_vertex_ai_str == "true"

    use_cache = kwargs.pop("use_client_cache", True)  # Allow disabling cache if needed

    google_client = await create_google_async_client(
        api_key=api_key,
        project_id=project_id,
        location=location,
        use_vertex_ai=use_vertex_ai,
        client_configs=client_configs,
        use_cache=use_cache,
        base_url=base_url,
        timeout=timeout,
    )

    if not model:
        model = os.environ.get("EMBEDDING_MODEL", DEFAULT_GOOGLE_EMBEDDING_MODEL)

    logger.debug(f"Requesting embeddings for {len(texts)} texts with model {model}.")
    verbose_debug(f"Embedding texts (first 3): {texts[:3]}")

    # Valid task types include: "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY", "SEMANTIC_SIMILARITY", "CLASSIFICATION", "CLUSTERING"
    task_type_str: Optional[str] = None
    if task_type:
        # Normalize the task type string to uppercase
        task_type_str = task_type.upper()
        logger.debug(f"Using task_type: {task_type_str}")

    # Set output_dimensionality to 1536 if not specified
    # gemini-embedding-001 outputs 3072 by default, but we want 1536 for consistency
    if output_dimensionality is None:
        output_dimensionality = DEFAULT_GOOGLE_EMBEDDING_DIM  # 1536

    try:
        response = await google_client.aio.models.embed_content(
            model=model,
            contents=texts,
            config=google_types.EmbedContentConfig(
                output_dimensionality=output_dimensionality,
                task_type=task_type_str,
            ),
        )

        if (
            not response
            or not hasattr(response, "embeddings")
            or not response.embeddings
        ):
            logger.error("Invalid or empty embedding response from Google API (batch).")
            raise InvalidResponseError(
                "Invalid or empty embedding response from Google API (batch)."
            )

        embeddings_list = [
            embedding_obj.values for embedding_obj in response.embeddings
        ]

        # Ensure all embeddings have the same dimension if output_dimensionality was not set,
        # or match output_dimensionality if it was.
        if embeddings_list and output_dimensionality:
            if any(len(emb) != output_dimensionality for emb in embeddings_list):
                logger.warning(
                    f"Some embeddings have dimension other than requested {output_dimensionality}. Check API behavior."
                )

        embeddings = np.array(embeddings_list, dtype=np.float32)

        # Apply L2 normalization for dimensions < 3072
        # The 3072 dimension embedding is already normalized by Gemini API
        if output_dimensionality and output_dimensionality < 3072:
            # Normalize each embedding vector to unit length
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Avoid division by zero
            norms = np.where(norms == 0, 1, norms)
            embeddings = embeddings / norms
            logger.debug(
                f"Applied L2 normalization to {len(embeddings)} embeddings of dimension {output_dimensionality}"
            )

        return embeddings

    except google_api_exceptions.GoogleAPIError as e:
        logger.error(f"Google API Error during embedding: {e.__class__.__name__} - {e}")
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error during Google embedding: {e.__class__.__name__} - {e}"
        )
        raise


@wrap_embedding_func_with_attrs(
    embedding_dim=DEFAULT_GOOGLE_EMBEDDING_DIM,
    max_token_size=DEFAULT_GOOGLE_MAX_TOKEN_SIZE,
)
async def google_embed_insert(
    texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT", **kwargs: Any
) -> np.ndarray:
    """
    Generates embeddings for a list of texts using Google's embedding models during insertion.
    """
    # Ensure output_dimensionality matches the wrapper's embedding_dim if not specified
    if "output_dimensionality" not in kwargs:
        kwargs["output_dimensionality"] = DEFAULT_GOOGLE_EMBEDDING_DIM  # 1536
    return await google_embed(texts=texts, task_type=task_type, **kwargs)
