"""General-purpose local-LLM client + PHI-specific redaction operations.

The `config`, `client`, `cache`, and `template` modules are PHI-agnostic and
usable by any lab script that wants to talk to a local LLM server over the
OpenAI-compatible chat-completions API. PHI-specific behavior lives in
sibling modules that build on top.
"""
