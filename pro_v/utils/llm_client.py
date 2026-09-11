"""
LLM Client Wrapper using LlamaIndex for OpenAI-compatible providers

This module provides a wrapper around LlamaIndex's OpenAI-compatible LLM
to connect to vLLM or hosted OpenAI-compatible endpoints with proper
configuration.
"""

import logging
import os
import random
from typing import List, Optional

from llama_index.llms.openai_like import OpenAILike

logger = logging.getLogger(__name__)


class VLLMLLMClient:
    """
    LLM Client wrapper for vLLM using LlamaIndex

    This client supports:
    - Multiple vLLM endpoints with round-robin load balancing
    - Configurable temperature and top_p
    - OpenAI-compatible API through LlamaIndex
    """

    def __init__(
        self,
        endpoints: List[str],
        model_name: str,
        temperature: float = 0.0,
        top_p: float = 0.1,
        max_tokens: int = 20000,
        timeout: float = 120.0
    ):
        """Initialize the LLM client

        Args:
            endpoints: List of endpoint URLs (e.g., ["http://127.0.0.1:8020"] or
                ["https://api.openai.com"])
            model_name: Model name served by the provider
            temperature: Sampling temperature (0.0 for greedy)
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        self.endpoints = endpoints
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.current_endpoint_idx = 0

        provider = os.getenv("LLM_PROVIDER", "vllm").strip().lower()
        api_key = os.getenv("OPENAI_API_KEY") if provider == "openai" else "dummy"
        configured_base = os.getenv("OPENAI_BASE_URL", "").strip()
        if provider == "openai" and not api_key:
            raise ValueError("OPENAI_API_KEY must be set when LLM_PROVIDER=openai")

        logger.info(f"VLLMLLMClient initialized:")
        logger.info(f"  - Provider: {provider}")
        logger.info(f"  - Endpoints: {endpoints}")
        logger.info(f"  - Model: {model_name}")
        logger.info(f"  - Temperature: {temperature}")
        logger.info(f"  - Top-p: {top_p}")
        logger.info(f"  - Max tokens: {max_tokens}")

        # Create LlamaIndex LLM instances for each endpoint
        self.llms = []
        for endpoint in endpoints:
            api_base = configured_base or endpoint
            api_base = api_base.rstrip("/")
            if provider != "openai" and not api_base.endswith("/v1"):
                api_base = f"{api_base}/v1"
            llm = OpenAILike(
                api_base=api_base,
                api_key=api_key,
                model=model_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                timeout=timeout,
                is_chat_model=True
            )
            self.llms.append(llm)
            logger.info(f"  - Created LLM client for: {api_base}")

    def _get_next_llm(self):
        """Get next LLM using round-robin load balancing"""
        llm = self.llms[self.current_endpoint_idx]
        self.current_endpoint_idx = (self.current_endpoint_idx + 1) % len(self.llms)
        return llm

    def chat(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Send a chat request to vLLM

        Args:
            system: System prompt
            user: User prompt
            temperature: Override default temperature (optional)
            top_p: Override default top_p (optional)
            max_tokens: Override default max_tokens (optional)

        Returns:
            Response text from the model
        """
        from llama_index.core.llms import ChatMessage, MessageRole

        # Get LLM with load balancing
        llm = self._get_next_llm()

        # Override parameters if provided
        if temperature is not None:
            llm.temperature = temperature
        if top_p is not None:
            llm.top_p = top_p
        if max_tokens is not None:
            llm.max_tokens = max_tokens

        # Prepare messages
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system),
            ChatMessage(role=MessageRole.USER, content=user)
        ]

        try:
            # Make chat request
            response = llm.chat(messages)
            return response.message.content

        except Exception as e:
            logger.error(f"LLM chat request failed: {e}")
            raise

    def complete(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Send a completion request to vLLM

        Args:
            prompt: Prompt text
            temperature: Override default temperature (optional)
            top_p: Override default top_p (optional)
            max_tokens: Override default max_tokens (optional)

        Returns:
            Completion text from the model
        """
        # Get LLM with load balancing
        llm = self._get_next_llm()

        # Override parameters if provided
        if temperature is not None:
            llm.temperature = temperature
        if top_p is not None:
            llm.top_p = top_p
        if max_tokens is not None:
            llm.max_tokens = max_tokens

        try:
            # Make completion request
            response = llm.complete(prompt)
            return response.text

        except Exception as e:
            logger.error(f"LLM completion request failed: {e}")
            raise


def create_llm_client_from_config(
    endpoints_csv: str,
    model_name: str,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> VLLMLLMClient:
    """Create LLM client from configuration

    This function reads configuration from environment variables
    and creates a VLLMLLMClient instance.

    Args:
        endpoints_csv: Comma-separated vLLM endpoints
        model_name: Model name
        temperature: Temperature (defaults to env TEMPERATURE or 0.0)
        top_p: Top-p (defaults to env TOP_P or 0.1)
        max_tokens: Max tokens (defaults to env MAX_TOKEN or 20000)

    Returns:
        Configured VLLMLLMClient instance
    """
    provider = os.getenv("LLM_PROVIDER", "vllm").strip().lower()
    if provider == "openai":
        endpoints = [os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")]
        model_name = os.getenv("OPENAI_MODEL", model_name)
    else:
        endpoints = [ep.strip() for ep in endpoints_csv.split(',') if ep.strip()]

    if not endpoints:
        raise ValueError("No LLM endpoints provided")

    # Get configuration from environment with fallbacks
    if temperature is None:
        temperature = float(os.getenv('TEMPERATURE', '0.0'))
    if top_p is None:
        top_p = float(os.getenv('TOP_P', '0.1'))
    if max_tokens is None:
        max_tokens = int(os.getenv('MAX_TOKEN', '20000'))

    return VLLMLLMClient(
        endpoints=endpoints,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )


def create_pychecker_llm_client_from_config(
    endpoints_csv: str,
    model_name: str,
    sample_temperature: Optional[float] = None,
    sample_top_p: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> VLLMLLMClient:
    """Create PyChecker-specific LLM client with sampling temperature

    PyChecker uses different temperature for sampling diversity.

    Args:
        endpoints_csv: Comma-separated vLLM endpoints
        model_name: Model name
        sample_temperature: Sampling temperature (defaults to env TEMPERATURE_SAMPLE or 0.6)
        sample_top_p: Sampling top-p (defaults to env TOP_P_SAMPLE or 0.95)
        max_tokens: Max tokens (defaults to env MAX_TOKEN or 20000)

    Returns:
        Configured VLLMLLMClient instance for PyChecker
    """
    provider = os.getenv("LLM_PROVIDER", "vllm").strip().lower()
    if provider == "openai":
        endpoints = [os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")]
        model_name = os.getenv("OPENAI_MODEL", model_name)
    else:
        endpoints = [ep.strip() for ep in endpoints_csv.split(',') if ep.strip()]

    if not endpoints:
        raise ValueError("No LLM endpoints provided")

    # Get configuration from environment with fallbacks
    if sample_temperature is None:
        sample_temperature = float(os.getenv('TEMPERATURE_SAMPLE', '0.6'))
    if sample_top_p is None:
        sample_top_p = float(os.getenv('TOP_P_SAMPLE', '0.95'))
    if max_tokens is None:
        max_tokens = int(os.getenv('MAX_TOKEN', '20000'))

    return VLLMLLMClient(
        endpoints=endpoints,
        model_name=model_name,
        temperature=sample_temperature,
        top_p=sample_top_p,
        max_tokens=max_tokens
    )
