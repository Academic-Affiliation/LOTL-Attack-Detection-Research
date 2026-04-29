"""
Eight deterministic obfuscation transforms used to generate the obfuscated
evaluation set (paper Section 4.1, n=250).

Technique mapping from paper:
  base64 encoding    (40 samples)
  string concat      (35 samples)
  backtick injection (30 samples)
  alias substitution (35 samples)
  case randomization (30 samples)
  char encoding      (25 samples)
  format operators   (30 samples)
  whitespace insert  (25 samples)
"""

import base64
import random
import re
import string

# PowerShell alias pairs documented in Volt Typhoon advisories (CISA/Microsoft)
_PS_ALIASES = {
    "Invoke-WebRequest": "iwr",
    "Get-Content":       "gc",
    "Set-Content":       "sc",
    "Get-ChildItem":     "gci",
    "Remove-Item":       "ri",
    "Write-Output":      "echo",
    "ForEach-Object":    "%",
    "Where-Object":      "?",
    "Select-Object":     "select",
    "Invoke-Expression": "iex",
}


def obfuscate_base64(command: str) -> str:
    """Encode full command payload as UTF-16LE base64 (PowerShell -EncodedCommand style)."""
    encoded = base64.b64encode(command.encode("utf-16-le")).decode("ascii")
    return f"powershell -EncodedCommand {encoded}"


def obfuscate_concatenation(command: str) -> str:
    """Split command into character-level string concatenation fragments."""
    mid = len(command) // 2
    left, right = command[:mid], command[mid:]
    return f'("{left}"' + "+" + f'"{right}")'


def obfuscate_backtick(command: str) -> str:
    """Insert PowerShell backtick escape characters into tokens (evades regex matching)."""
    words = command.split()
    result = []
    for i, word in enumerate(words):
        if i % 2 == 0 and len(word) > 3:
            mid = len(word) // 2
            word = word[:mid] + "`" + word[mid:]
        result.append(word)
    return " ".join(result)


def obfuscate_alias(command: str) -> str:
    """Substitute known PowerShell cmdlet names with documented short aliases."""
    result = command
    for full, alias in _PS_ALIASES.items():
        result = result.replace(full, alias)
    return result


def obfuscate_case(command: str, seed: int = 42) -> str:
    """Randomise character case (evades case-sensitive signature matching)."""
    rng = random.Random(seed)
    return "".join(c.upper() if rng.random() > 0.5 else c.lower() for c in command)


def obfuscate_char_encoding(command: str) -> str:
    """Replace alphanumeric characters with hex escape sequences."""
    result = []
    for ch in command:
        if ch.isalnum():
            result.append(f"[char]0x{ord(ch):02x}")
        else:
            result.append(ch)
    return "".join(result)


def obfuscate_format_op(command: str) -> str:
    """Use PowerShell format operator to reorder string fragments."""
    words = command.split()
    if len(words) < 2:
        return command
    indices = list(range(len(words)))
    fmt_str = " ".join(f"{{{i}}}" for i in indices)
    args = ",".join(f'"{w}"' for w in words)
    return f'("{fmt_str}" -f {args})'


def obfuscate_whitespace(command: str) -> str:
    """Insert extra spaces and tab characters between tokens."""
    return re.sub(r" ", "  ", command)


TECHNIQUE_MAP = {
    "base64":        obfuscate_base64,
    "concatenation": obfuscate_concatenation,
    "backtick":      obfuscate_backtick,
    "alias":         obfuscate_alias,
    "case":          obfuscate_case,
    "char_encoding": obfuscate_char_encoding,
    "format_op":     obfuscate_format_op,
    "whitespace":    obfuscate_whitespace,
}


def apply(command: str, technique: str) -> str:
    if technique not in TECHNIQUE_MAP:
        raise ValueError(f"Unknown technique '{technique}'. Valid: {list(TECHNIQUE_MAP)}")
    return TECHNIQUE_MAP[technique](command)
