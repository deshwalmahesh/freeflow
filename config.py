"""
Free LLM Router Configuration

This module contains configuration for various AI model providers and their available models.
Providers are sorted by preference based on model quality and availability.
"""

# Common rate limit error messages across different providers
RATE_LIMIT_SYNS = (
    "429",
    "too many requests",
    "rate",
    "quota",
    "exhausted",
    "over quota",
    "ratelimiterror",
    "resource_exhausted"
)


PROVIDER_CONFIG = [
    {
        "provider": "GOOGLE_AI_STUDIO", # Best Models: Free models with 10-RPM, 250K-TPM
        "available_free_models": [
            "gemini-2.5-flash-preview-05-20",
            "gemini-2.0-flash"
        ]
    },
    
    {
        "provider": "GROQ",  # Ultra-low latency + Multimodality
        "available_free_models": [
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "llama-3.3-70b-versatile"
        ]
    },

    {
      "provider": "FIREWORKSAI", 
      "available_free_models": [
          "accounts/fireworks/models/llama4-maverick-instruct-basic",
          "accounts/fireworks/models/deepseek-r1-0528",
          "accounts/fireworks/models/qwen3-235b-a22b"
      ]
    },
    
    {
        "provider": "CEREBRAS",
        "available_free_models": [
            "llama-4-scout-17b-16e-instruct",
            "deepseek-r1-distill-llama-70b"
        ]
    },
    
    {
        "provider": "OPENROUTER", # Tight cap on requests and tokens per minute
        "available_free_models": [
            "meta-llama/llama-4-maverick:free",
            "qwen/qwen3-235b-a22b:free",
            "deepseek/deepseek-chat-v3-0324:free"
        ]
    },

        
    {
        "provider": "MISTRAL", # Decent models with 60 RPM
        "available_free_models": [
            "magistral-small-2506",
            "mistral-small-2503"
        ]
    },

    {
        "provider": "TOGETHERAI", # They seem to have $1 free credit for new users not sure how does it really work though
        "available_free_models": [
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            "Qwen/Qwen3-235B-A22B-fp8-tput",
            "deepseek-ai/DeepSeek-V3"
        ]
    }
]