"""The packages share the `explorers` namespace: each must import next to the others."""


def test_all_packages_import_together():
    import explorers.core
    import explorers.learning
    import explorers.populations
    from explorers.populations.episode import Episode

    assert explorers.core.Examples and explorers.learning.toy and Episode
    assert not hasattr(__import__("explorers"), "__file__") or __import__("explorers").__file__ is None
