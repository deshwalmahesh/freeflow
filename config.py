# Common rate limit error messages across different providers
RATE_LIMIT_SYNS = (
    "429", "too many requests", "rate", "quota", "exhausted",
    "over quota", "ratelimiterror", "resource_exhausted"
)

# This list is sorted based on some criteria such as BEST Models and their availability
PROVIDER_CONFIG = [
  {
      "provider": "GOOGLE_AI_STUDIO", # Best Models: Free models with 10-RPM, 250K-TPM
      "available_free_models": [
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.0-flash"
      ],
      "default_model_to_use": "gemini-2.5-flash-preview-05-20"
    },

    {
      "provider": "FIREWORKSAI", # Highest open-weight accuracy on benchmarks
      "available_free_models": [
        "accounts/fireworks/models/llama-v3p1-405b-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "accounts/fireworks/models/llama-3.3-70b-instruct",
        "deepseek-ai/deepseek-r1-70b"
      ],
      "default_model_to_use": "accounts/fireworks/models/llama-v3p1-405b-instruct"
    },
    {
      "provider": "GROQ", # Ultra-low latency
      "available_free_models": [
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "llama-3.3-70b-versatile",
        "deepseek-r1-distill-llama-70b"
      ],
      "default_model_to_use": "meta-llama/llama-4-maverick-17b-128e-instruct"
    },
    {
      "provider": "TOGETHERAI", # Good accuracy, moderate limits
      "available_free_models": [
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "deepseek-ai/DeepSeek-V3"
      ],
      "default_model_to_use": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
    },
    {
      "provider": "MISTRAL", # Great quality but 1 RPS
      "available_free_models": [
        "mistral-large-latest",
        "mistral-medium-3",
        "mistral-small-24b-instruct"
      ],
      "default_model_to_use": "mistral-large-latest"
    },
    {
        "provider": "OPENROUTER", # Tight Cap on Requests and Tokens per minutes
        "available_free_models": [
            "mistralai/mixtral-8x22b-instruct",
            "microsoft/wizardlm-2-8x22b:nitro:free",
            "deepseek-ai/deepseek-r1:free"
        ],
        "default_model_to_use": "mistralai/mixtral-8x22b-instruct"
    },
    {
        "provider": "CEREBRAS", # Nearly unlimited tokens, variable latency
        "available_free_models": [
            "llama-3.3-70b",
            "llama-4-scout-17b-16e-instruct"
        ],
        "default_model_to_use": "llama-3.3-70b"
    }
  ]