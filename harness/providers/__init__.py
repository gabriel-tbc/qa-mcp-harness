"""Provider layer: the model-agnostic boundary.

Everything model-specific lives here, behind the `Provider` protocol and the
neutral `ToolSpec` / `ModelResponse` types defined in `base`. The rest of the
harness speaks only those neutral types and never imports a vendor SDK.
"""
