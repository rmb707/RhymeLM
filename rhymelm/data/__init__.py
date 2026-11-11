"""Data utilities. Exports resolve lazily so light tools (ingest, phonemes)
don't pull in torch or pandas just by importing a submodule."""

_EXPORTS = {
    "build_dual_corpus": "rhymelm.data.corpus",
    "extract_verses": "rhymelm.data.corpus",
    "load_dictionary": "rhymelm.data.corpus",
    "load_lyrics": "rhymelm.data.corpus",
    "CharTokenizer": "rhymelm.data.tokenizer",
    "LMDataset": "rhymelm.data.dataset",
    "create_dataloaders": "rhymelm.data.dataset",
    "BPETokenizer": "rhymelm.data.bpe",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        import importlib
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'rhymelm.data' has no attribute {name!r}")
