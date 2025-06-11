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
from langchain_core.language_models.chat_models import BaseChatModel

# Handle both relative and absolute imports
try:
    from .config import PROVIDER_CONFIG, RATE_LIMIT_SYNS
except ImportError:
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
                models.append(ChatOpenAI(model=model_name, api_key=api_key, base_url="https://api.openrouter.ai/v1",
                max_tokens=6665))
            elif provider == "CEREBRAS":
                models.append(ChatOpenAI(model=model_name,api_key=api_key, base_url="https://api.cerebras.ai/v1",))
            else:
                logger.warning(f"Unknown provider: {provider}")
                break

            models[-1].__dict__["api_key_index"] = api_key_env
        except Exception as e:
            logger.error(f"Error initializing {provider} with key {idx}: {type(e).__name__}: {e}")
        
        idx += 1
    
    return models

class FreeFlowRouter(BaseChatModel):
    """
    Router that delegates ANY method call to the best available LangChain model.
    
    Simple logic:
    1. Load all models from config
    2. Best model = last successful model, or first available (index 0+)
    3. Any method call gets delegated with automatic failover
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        self._config = config if config is not None else PROVIDER_CONFIG
        self._llms = []
        self._backoff_until = []
        self._last_successful_idx = 0
        self._init_llms()
        
    def _init_llms(self):
        """Load all models from config."""
        self._llms = []
        self._backoff_until = []
        
        for provider_config in self._config:
            provider = provider_config["provider"]
            model = provider_config["available_free_models"][0]
            
            models = _build_llm(provider, model)
            if models:
                self._llms.extend(models)
                self._backoff_until.extend([0] * len(models))
                logger.info(f"Loaded {provider} with {model} ({len(models)} keys)")
            else:
                logger.warning(f"Failed to load {provider} with {model}")

        if not self._llms:
            raise RuntimeError("No providers initialized - check API keys in .env file")
    
    def _get_best_model(self):
        """Get the best available model: last successful, or first available."""
        if not self._llms:
            raise RuntimeError("No LLM providers available")
            
        current_time = time.time()
        
        # Try last successful model first
        if (self._last_successful_idx < len(self._llms) and 
            current_time >= self._backoff_until[self._last_successful_idx]):
            return self._llms[self._last_successful_idx]
        
        # Otherwise, find first available (index 0+)
        for idx in range(len(self._llms)):
            if current_time >= self._backoff_until[idx]:
                return self._llms[idx]
        
        # All in backoff - return shortest backoff
        min_idx = min(range(len(self._backoff_until)), key=lambda i: self._backoff_until[i])
        return self._llms[min_idx]
    
    def _apply_backoff(self, idx, error_msg):
        """Apply backoff based on error type."""
        current_time = time.time()
        if any(s in error_msg.lower() for s in RATE_LIMIT_SYNS):
            # Exponential backoff for rate limits
            current_backoff = max(60, (self._backoff_until[idx] - current_time) * 2 
                                if self._backoff_until[idx] > current_time else 60)
            self._backoff_until[idx] = current_time + current_backoff
        else:
            # Short backoff for other errors
            self._backoff_until[idx] = current_time + 10
    
    def _try_all_models(self, method_name, *args, **kwargs):
        """Try method on all models with smart ordering."""
        current_time = time.time()
        last_exception = None
        
        # Smart ordering: last successful first, then others
        indices = []
        if (self._last_successful_idx < len(self._llms) and 
            current_time >= self._backoff_until[self._last_successful_idx]):
            indices.append(self._last_successful_idx)
        
        for idx in range(len(self._llms)):
            if idx != self._last_successful_idx and current_time >= self._backoff_until[idx]:
                indices.append(idx)
        
        if not indices:  # All in backoff
            indices = sorted(range(len(self._llms)), key=lambda i: self._backoff_until[i])
        
        # Try each model
        for idx in indices:
            llm = self._llms[idx]
            try:
                if method_name == '_generate':
                    result = llm._generate(*args, **kwargs)
                else:
                    method = getattr(llm, method_name)
                    result = method(*args, **kwargs) if callable(method) else method
                
                # Success!
                self._backoff_until[idx] = 0
                self._last_successful_idx = idx
                return result
                
            except AttributeError:
                continue  # Model doesn't have this method
            except Exception as err:
                last_exception = err
                self._apply_backoff(idx, str(err))
                continue
        
        # All failed
        if last_exception:
            raise last_exception
        else:
            raise AttributeError(f"No model supports method '{method_name}'")

    def _try_structured_output(self, schema, input, **kwargs):
        """Try structured output on all models with smart ordering."""
        current_time = time.time()
        last_exception = None
        
        # Smart ordering: last successful first, then others
        indices = []
        if (self._last_successful_idx < len(self._llms) and 
            current_time >= self._backoff_until[self._last_successful_idx]):
            indices.append(self._last_successful_idx)
        
        for idx in range(len(self._llms)):
            if idx != self._last_successful_idx and current_time >= self._backoff_until[idx]:
                indices.append(idx)
        
        if not indices:  # All in backoff
            indices = sorted(range(len(self._llms)), key=lambda i: self._backoff_until[i])
        
        # Try each model
        for idx in indices:
            llm = self._llms[idx]
            try:
                structured_llm = llm.with_structured_output(schema, **kwargs)
                result = structured_llm.invoke(input)
                
                # Success!
                self._backoff_until[idx] = 0
                self._last_successful_idx = idx
                return result
                
            except (AttributeError, NotImplementedError):
                continue  # Model doesn't support structured output
            except Exception as err:
                last_exception = err
                self._apply_backoff(idx, str(err))
                continue
        
        # All failed
        if last_exception:
            raise last_exception
        else:
            raise AttributeError("No model supports with_structured_output functionality")

    # Required BaseChatModel methods
    @property
    def _llm_type(self) -> str:
        return "FreeFlowRouter"
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._try_all_models('_generate', messages, stop=stop, 
                                  run_manager=run_manager, **kwargs)
    
    def bind_tools(self, tools, **kwargs):
        """Bind tools to all underlying models in-place."""
        new_llms = []
        
        for idx, llm in enumerate(self._llms):
            try:
                bound_llm = llm.bind_tools(tools, **kwargs)
                new_llms.append(bound_llm)
            except (NotImplementedError, AttributeError):
                # Skip models that don't support bind_tools
                continue
        
        if not new_llms:
            raise RuntimeError("No models support bind_tools functionality")
        
        # Replace with only the models that support tools
        self._llms = new_llms.copy()
        
        # Reset exponential backoff since we're binding tools at the beginning
        self._backoff_until = [0] * len(self._llms)
        self._last_successful_idx = 0
        
        return self
    
    def with_structured_output(self, schema, **kwargs):
        """Return a wrapper that uses structured output with failover."""
        class StructuredOutputWrapper:
            def __init__(self, router, schema, **kwargs):
                self.router = router
                self.schema = schema
                self.kwargs = kwargs
            
            def invoke(self, input, **invoke_kwargs):
                # Use the existing _try_structured_output method with failover
                return self.router._try_structured_output(self.schema, input, **{**self.kwargs, **invoke_kwargs})
            
            def __getattr__(self, attr_name):
                # For any other method, delegate to the first working structured model
                def wrapper_method(*args, **kwargs):
                    structured_model = self.router._try_all_models('with_structured_output', self.schema, **self.kwargs)
                    return getattr(structured_model, attr_name)(*args, **kwargs)
                return wrapper_method
        
        return StructuredOutputWrapper(self, schema, **kwargs)
    
    # Convenience properties
    @property
    def active_model(self):
        return self._get_best_model()
    
    @property
    def model_name(self):
        active = self._get_best_model()
        return getattr(active, 'model_name', getattr(active, 'model', 'unknown'))
    
    @property 
    def model(self):
        return self.model_name
    
    def get_model_info(self):
        """Get info about current active model."""
        active = self._get_best_model()
        return {
            'class_name': active.__class__.__name__,
            'model_name': getattr(active, 'model_name', getattr(active, 'model', 'unknown')),
            'api_key_index': getattr(active, 'api_key_index', 'unknown'),
            'provider_count': len(self._llms),
            'active_provider_index': self._llms.index(active) if active in self._llms else -1
        }
    
    def invoke_with_failover(self, *args, **kwargs):
        """Explicit failover invoke - same as regular invoke()."""
        return self.invoke(*args, **kwargs)
    
    def invoke_active_only(self, *args, **kwargs):
        """Invoke using only the active model without failover."""
        return self.active_model.invoke(*args, **kwargs)
    
    def __getattr__(self, name):
        """Delegate ANY method to best model with failover."""
        # Skip internal attributes
        if name.startswith('_') or name.startswith('__pydantic'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        # Special case for LangChain's 'get' method
        if name == 'get':
            def get_method(key, default=None):
                if key == 'callback_manager':
                    return None
                return getattr(self._get_best_model(), key, default)
            return get_method
        
        # For any other method/attribute - try with failover
        def method_with_failover(*args, **kwargs):
            return self._try_all_models(name, *args, **kwargs)
        
        # Check if any model has this attribute
        for llm in self._llms:
            try:
                attr = getattr(llm, name)
                return method_with_failover if callable(attr) else attr
            except AttributeError:
                continue
        
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


if __name__ == "__main__":
    # Handle both relative and absolute imports
    try:
        from .config import PROVIDER_CONFIG
    except ImportError:
        from config import PROVIDER_CONFIG
    
    router = FreeFlowRouter(config=PROVIDER_CONFIG)
    
    print("FreeFlowRouter - Clean & Simple")
    print(f"Loaded {len(router._llms)} models")
    print(f"Active: {router.get_model_info()}")
    
    # Test basic functionality
    response = router.invoke("Hello!")
    print(f"Response: {response.content}")
    
    # Test any LangChain method - they all work with failover!
    print(f"Model name: {router.model_name}")
    
    print("\nAll LangChain methods work with automatic failover!")
