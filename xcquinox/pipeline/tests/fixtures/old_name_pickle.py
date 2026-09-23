"""A pickle that names the package's classes under the subpackage's old name.

The cluster harness pickles each spec beside its checkpoint, and every spec the campaigns
stored under the old module path names the classes there (the
first element of ``checkpoint_class.RENAMED_PACKAGE``); the tests of the readers that must
keep loading them make such a pickle from a live object.
"""
import pickle


def pickled_under_the_old_name(obj) -> bytes:
    """``obj`` pickled under protocol 2, whose GLOBAL opcode writes a class's module as
    newline-terminated text with no length prefix and no frame, so every class path of the
    renamed package is rewritten to the old name by a plain byte replacement (a protocol-5
    pickle carries frame lengths the replacement would leave wrong)."""
    from xcquinox.pipeline import checkpoint_class as cc
    old, new = cc.RENAMED_PACKAGE
    data = pickle.dumps(obj, protocol=2)
    patched = data.replace(new.encode(), old.encode())
    assert patched != data, "the pickle names no class of the package"
    return patched
