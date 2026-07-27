import os
from dotenv import load_dotenv
from groq import Groq
from google import genai

from config import PROVIDER, MODELS

load_dotenv()

# -------------------------------------------------
# Load API keys
# -------------------------------------------------
# Groq clients are loaded UNCONDITIONALLY (not just when PROVIDER=="groq"):
# reflection.py's teacher model always runs on Groq (llama-3.3-70b-versatile)
# regardless of which provider the main agent loop is using, so the key
# pool and rotation state need to exist even when PROVIDER == "gemini".
# Previously reflection.py built its own separate, single, non-rotating
# Groq client - if that one key hit a rate limit, reflection silently
# stopped working for the rest of the run even though the main agent path
# had failover via this module.

groq_clients = []
gemini_clients = []

current_groq = 0
current_gemini = 0

i = 1
while True:
    key = os.getenv(f"GROQ_API_KEY_{i}")
    if not key:
        break

    groq_clients.append(Groq(api_key=key))
    i += 1

# Backward compatibility
if not groq_clients:
    key = os.getenv("GROQ_API_KEY")
    if key:
        groq_clients.append(Groq(api_key=key))

if not groq_clients:
    raise ValueError(
        "No Groq API key found. Set GROQ_API_KEY or GROQ_API_KEY_1... "
        "(required even if PROVIDER != 'groq', since reflection.py's "
        "teacher model always runs on Groq)."
    )


if PROVIDER == "gemini":

    i = 1
    while True:
        key = os.getenv(f"GEMINI_API_KEY_{i}")
        if not key:
            break

        gemini_clients.append(genai.Client(api_key=key))
        i += 1

    # Backward compatibility
    if not gemini_clients:
        key = os.getenv("GEMINI_API_KEY")
        if key:
            gemini_clients.append(genai.Client(api_key=key))

    if not gemini_clients:
        raise ValueError(
            "No Gemini API key found. Set GEMINI_API_KEY or GEMINI_API_KEY_1..."
        )


# -------------------------------------------------
# Shared Groq call + key rotation
# -------------------------------------------------
# Used both by generate() below (when PROVIDER == "groq") and by
# generate_groq() (called directly by reflection.py's teacher, which needs
# Groq regardless of the main PROVIDER setting). Centralizing this means
# both call sites get identical multi-key rotation/failover behavior.
def _call_groq(model, messages, temperature=0, max_tokens=1500, response_format=None):
    global current_groq

    last_error = None

    for _ in range(len(groq_clients)):

        client = groq_clients[current_groq]

        try:
            kwargs = dict(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
            if response_format is not None:
                kwargs["response_format"] = response_format

            response = client.chat.completions.create(**kwargs)

            return response.choices[0].message.content.strip()

        except Exception as e:

            last_error = e
            error = str(e).lower()

            # Rotate only for quota/rate-limit errors
            if any(x in error for x in (
                "429",
                "rate limit",
                "rate_limit_exceeded",
                "tokens per day",
                "quota",
                "resource exhausted",

            )):

                print(
                    f"[!] Groq key {current_groq + 1} exhausted. "
                    "Switching to next key..."
                )

                current_groq = (
                    current_groq + 1
                ) % len(groq_clients)

                print(
                    f"[+] Now using Groq key {current_groq + 1}"
                )

                continue

            raise

    raise last_error


def generate_groq(messages, model, temperature=0, max_tokens=1500, response_format=None):
    """
    Public entry point for callers that need Groq directly with an
    explicit model, independent of config.PROVIDER (e.g. reflection.py's
    teacher, which always uses llama-3.3-70b-versatile even when the main
    agent loop is on a different provider). Shares the same multi-key
    rotation as generate().
    """
    return _call_groq(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
    )


# -------------------------------------------------
# Generic interface
# -------------------------------------------------

def generate(messages, temperature=0, max_tokens=1500):

    global current_gemini

    # =====================================================
    # GROQ
    # =====================================================

    if PROVIDER == "groq":
        return _call_groq(
            model=MODELS[PROVIDER],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # =====================================================
    # GEMINI
    # =====================================================

    elif PROVIDER == "gemini":

        last_error = None

        for _ in range(len(gemini_clients)):

            client = gemini_clients[current_gemini]

            try:

                prompt = (
                    messages[0]["content"]
                    + "\n\n"
                    + messages[1]["content"]
                )

                response = client.models.generate_content(
                    model=MODELS[PROVIDER],
                    contents=prompt,
                )

                return response.text.strip()

            except Exception as e:

                last_error = e
                error = str(e).lower()

                if any(x in error for x in (
                    "429",
                    "resource exhausted",
                    "quota",
                    "rate limit",
                )):

                    print(
                        f"[!] Gemini key {current_gemini + 1} exhausted. "
                        "Switching to next key..."
                    )

                    current_gemini = (
                        current_gemini + 1
                    ) % len(gemini_clients)

                    continue

                raise

        raise last_error

    raise ValueError(f"Unsupported provider: {PROVIDER}")
