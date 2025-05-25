# 🚀 Free Flow: Automatic Router for Free + Best LLMs

> **Winner of the "Best tool for poor AI developers" award: 2025**


## 🤔 Why I Built This

Ever got the idea for an amazing AI wrapper app which can make you billionaire only to realise that your beloved AI provider sends this lovely message...


```
ERROR: You again man!! Seriously? I told you, your limit had already been exceeded. This time, try again in 3 years, 7 months. FO!!
```

$${\color{red}\text{Aakhir aa hi gaya na aukaat pe!!!}}$$

So after the 173rd time this had happened to me while trying to get an AI to explain why I'm poor (spoiler: I forgot to be born in a rich family), I decided enough was enough. 

I built this router because:

1. I'm too ~~cheap~~ financially responsible to pay for multiple AI subscriptions
2. I wanted the power of 70B+ parameter models without selling a kidney
3. I believe in AI democracy - powerful models for all!
4. I got tired of playing "API key roulette" with 7 different services

## 🎯 What This Does

The Free LLM Router is your wingman that:

- Automatically routes your requests to the best available free LLM provider
- Handles rate limits and failures with grace (unlike me when my code doesn't compile)
- Supports structured output for when you need your AI to be more organized than your sock drawer
- Works with the biggest, baddest models available for free (ATLEAST 13B+ parameters)
- Makes you look like an AI wizard to your friends

## 🛠️ How to Use It

### Basic Usage

```python
from free_llm_router import OpenRouter

router = OpenRouter()
response = router.invoke("Tell me a joke about misery")
print(response)

>> "Your life bruv!!!"
```

### For Structured Output

```python
from pydantic import BaseModel, Field
from typing import List

class JokeBuilder(BaseModel):
    setup: str = Field(description="The setup line of the joke")
    punchline: str = Field(description="The punchline that makes it funny")

# Method 1: With wrapper
joke = router.with_structured_outputs(JokeBuilder).invoke("Tell me a joke about life")

# Method 2: Direct schema
joke = router.invoke(
    "Tell me a joke about life",
    schema=JokeBuilder
)
```

## 🔑 Getting Started

### 1. Install the Requirements

```bash
pip install -r requirements.txt
```

### 2. Get Your Free API Keys

Create accounts at these services to get free API keys:

- [Google AI Studio](https://makersuite.google.com/) - Gemini models
- [Fireworks AI](https://fireworks.ai/) - Llama models
- [Groq](https://console.groq.com/) - Ultra-fast inference
- [Together AI](https://www.together.ai/) - Various open models
- [Mistral AI](https://console.mistral.ai/) - Mistral models
- [OpenRouter](https://openrouter.ai/) - Gateway to many models
- [Cerebras](https://www.cerebras.ai/) - High-throughput models

### 2. Get Your Free API Keys

Create accounts at these services to get free API keys:

- [Google AI Studio](https://aistudio.google.com/)
- [Fireworks AI](https://fireworks.ai/)
- [Groq](https://console.groq.com/)
- [Together AI](https://www.together.ai/)
- [Mistral AI](https://console.mistral.ai/)
- [OpenRouter](https://openrouter.ai/)
- [Cerebras](https://www.cerebras.ai/)


Please, for the love of some intelligence, create a `.env` file with **YOUR** API keys and *DON'T* use placeholder keys like given below


Don't worry if you don't have all of them - the router will work with whatever keys you provide!

## 🧠 How It Works

1. The router tries to use the first available LLM in the priority list
2. If a provider hits rate limits (or any other error), it applies exponential backoff (fancy way of saying "wait longer each time")
3. It automatically falls back to the next provider if one fails
4. It keeps trying until it gets a response or runs out of options

## 🚫 Common Error Messages and What They Mean

- **"All LLM providers failed"**: You either have no valid API keys or all providers are rate-limited. You can't outrun your luck forever!
- **"No API key found for..."**: You're a dumb fk so add the missing API key to your `.env` file
- **"429 Too Many Requests"**: You're being too greedy! The provider needs a break

## 🤝 Contributing

Found a new free LLM provider? Want to improve the router? PRs welcome!

1. Fork the repo
2. Add your amazing improvements. It could be a new model, new provider, bug fix or even a new feature
3. Submit a PR
4. Wait while I procrastinate reviewing it

## ⚠️ Disclaimer

This project is for educational purposes only. Please respect the terms of service of all providers. I'm not responsible if your AI starts writing poetry instead of code, becomes sentient, or decides to take over the world or for the least, you get blocked, again!!!

## 📜 License

MIT License - Use it, abuse it, but please credit it.

---

> *Remember: The best things in life are free... including Billions of parameter language models!*

(and if they're not, work with me and let's make them)
