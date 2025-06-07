import os
# ChatFireworks breaks protobuf, so we set this environment variable
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import time
import logging
from typing import Optional, Type
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_together import ChatTogether
from langchain_fireworks import ChatFireworks
from langchain_mistralai import ChatMistralAI
from config import PROVIDER_CONFIG, RATE_LIMIT_SYNS




# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Changed from INFO to WARNING to reduce verbosity
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv(override=True)  # Loads '{PROVIDER_NAME}_API_KEY_{idx}' env-vars and forces reload
# ───────────────────────── Builder for one provider ─────────────────────────
def _build_llm(provider: str, model_name: str):
    """
    Build language model clients for the specified provider and model.
    
    Args:
        provider (str): The name of the LLM provider
        model_name (str): The specific model to use from the provider
        
    Returns:
        A list of langchain chat model instances or empty list if all API keys are missing
    """
    provider = provider.upper().strip()
    models = []
    idx = 1
    
    while True:
        api_key_env = f"{provider}_API_KEY_{idx}"
        api_key = os.environ.get(api_key_env)
        
        # If no API key found for this index, stop looking for more keys
        if not api_key:
            logger.debug(f"No API key found for {api_key_env}, stopping search for {provider} keys")
            break
            
        logger.debug(f"Found API key for {provider} (index {idx})")
        try:
            if provider == "FIREWORKSAI":
                models.append(ChatFireworks(model_name=model_name, fireworks_api_key=api_key))
            elif provider == "TOGETHERAI":
                models.append(ChatTogether(model_name=model_name, together_api_key=api_key))
            elif provider == "GOOGLE_AI_STUDIO":
                models.append(ChatGoogleGenerativeAI(model=model_name, api_key=api_key))
            elif provider == "GROQ":
                models.append(ChatGroq(model_name=model_name, groq_api_key=api_key))
            elif provider == "MISTRAL":
                models.append(ChatMistralAI(model=model_name, api_key=api_key))
            elif provider == "OPENROUTER":
                models.append(ChatOpenAI(model=model_name, api_key=api_key, base_url="https://openrouter.ai/api/v1",
                max_tokens=6665))
            elif provider == "CEREBRAS":
                models.append(ChatOpenAI(model=model_name,api_key=api_key, base_url="https://api.cerebras.ai/v1",))
            else:
                logger.warning(f"Unknown provider: {provider}")
                break
        except Exception as e:
            logger.error(f"Error initializing {provider} with key {idx}: {type(e).__name__}: {e}")
        
        idx += 1
    
    return models

# ────────────────────────── the router itself ───────────────────────────────
class OpenRouter:
    """
    Router for LLM providers that automatically handles failover.
    
    This router always starts with the first LLM in the list (llms[0]) and falls through 
    to other providers when encountering rate limits or other errors. It uses exponential 
    backoff to avoid repeatedly hitting rate limits on the same provider.
    """

    def __init__(self, config=None):
        """Initialize the router with backoff tracking for each LLM.
        
        Args:
            config (List[Dict[str, Any]], optional): Configuration for LLM providers.
                Defaults to MODELS_CONFIG.
        """
        self._config = config if config is not None else MODELS_CONFIG
        self._llms = []
        self._backoff_until = []
        self._init_llms()
        
    def _init_llms(self):
        """Initialize LLM providers from configuration."""
        self._llms = []
        self._backoff_until = []
        
        for provider_config in self._config:
            provider = provider_config["provider"]
            model = provider_config["default_model_to_use"]
            
            models = _build_llm(provider, model)
            if models:
                self._llms.extend(models)
                self._backoff_until.extend([0] * len(models))  # No backoff initially
                logger.info(f"Initialized {provider} with model {model} ({len(models)} API keys)")
            else:
                logger.warning(f"Failed to initialize {provider} with model {model} (no valid API keys)")

                
        if not self._llms:
            logger.critical("No providers initialized - check API keys in .env file")
            raise RuntimeError("No providers initialized - check API keys in .env file.")
        
    def _call(
        self,
        llm,
        prompt: str,
        schema: Optional[Type] = None,
        **kwargs,
    ):
        """
        Internal method to call an LLM with the given parameters.
        
        Args:
            llm: The language model to call
            prompt (str): The prompt to send to the LLM
            schema (Optional[Type]): Pydantic schema for structured output
            **kwargs: Additional arguments to pass to the LLM
            
        Returns:
            The response from the LLM
        """
        if schema:
            llm = llm.with_structured_output(schema)
        return llm.invoke(prompt, **kwargs)

    def invoke(
        self,
        prompt: str,
        schema: Optional[Type] = None,
        temperature: float = None,
        **kwargs,
    ):
        """
        Invoke an LLM with the given prompt and parameters.
        
        This method tries each LLM in the rotation list, starting from the first one
        and using exponential backoff for rate-limited models. It always tries to use
        the best models first if they're not in backoff.
        
        Args:
            prompt (str): The prompt to send to the LLM
            schema (Optional[Type]): Pydantic schema for structured output
            temperature (float, optional): Temperature parameter for generation
            **kwargs: Additional arguments to pass to the LLM
            
        Returns:
            The response from the first successful LLM
            
        Raises:
            RuntimeError: If all LLMs fail
        """
        # Get current time to check backoff status
        current_time = time.time()
        
        # Create a list of indices to try in original order (best first)
        indices_to_try = list(range(len(self._llms)))
            
        logger.debug(f"Starting LLM invocation with {len(indices_to_try)} providers available")
        
        for idx in indices_to_try:
            # Skip this LLM if it's in backoff period
            if current_time < self._backoff_until[idx]:
                backoff_remaining = int(self._backoff_until[idx] - current_time)
                logger.debug(f"Skipping provider at index {idx} (in backoff for {backoff_remaining} more seconds)")
                continue
                
            llm = self._llms[idx]
            provider_name = f"{llm.__class__.__name__} || Model: {llm.model_name if hasattr(llm, 'model_name') else 'UNKNOWN'}"
            
            logger.debug(f"Trying provider: {provider_name} (index {idx})")
            
            try:
                result = self._call(
                    llm,
                    prompt,
                    schema=schema,
                    temperature=temperature,
                    **kwargs,
                )
                # Success! Clear any backoff for this LLM
                self._backoff_until[idx] = 0
                logger.info(f"Successfully used {provider_name} (index {idx})")
                return result
                
            except Exception as err:
                error_msg = str(err).lower()
                  
                if any(s in error_msg for s in RATE_LIMIT_SYNS):
                    # Apply exponential backoff - start with 60s, double each time
                    # If this is the first failure, backoff 60s, otherwise double the current backoff
                    current_backoff = max(60, (self._backoff_until[idx] - current_time) * 2 if self._backoff_until[idx] > current_time else 60)
                    self._backoff_until[idx] = current_time + current_backoff
                    
                    logger.warning(
                        f"{provider_name} rate-limited "
                        f"(index {idx}) - backing off for {current_backoff}s, trying next provider."
                    )
                else:
                    # For non-rate-limit errors, use a shorter backoff
                    self._backoff_until[idx] = current_time + 10
                    # Log the actual exception type and message
                    logger.error(f"{provider_name} failed (index {idx}): {type(err).__name__}: {err}")
                continue
                
        logger.critical("All LLM providers failed")
        raise RuntimeError("All LLM providers failed - check logs for details")

    def with_structured_outputs(self, schema: Type):
        """
        Create a wrapper that always uses the specified schema for structured output.
        
        Args:
            schema (Type): Pydantic schema to use for structured output
            
        Returns:
            A wrapper object with an invoke method that uses the schema
        """
        class _Wrapper:
            def __init__(self, outer): 
                self._outer = outer
                
            def invoke(self, prompt: str, **kw):
                """Invoke with the predefined schema"""
                return self._outer.invoke(prompt, schema=schema, **kw)
                
        return _Wrapper(self)


if __name__ == "__main__":

    from core import OpenRouter
    from config import PROVIDER_CONFIG
    
    router = OpenRouter(config=PROVIDER_CONFIG)
    response = router.invoke("HELLO") # Best way to use as it assigns active model based on availability
    print(response)

    for llm in router._llms: # Each llm is a Langchain Model so you can directly use it as .invoke(), chain etc
        try:
            llm.invoke("Hello!")
        except Exception as e:
            print(F"Error occured:{e} for llm: {llm.__class__.__name__}")
            print("-"*100)
