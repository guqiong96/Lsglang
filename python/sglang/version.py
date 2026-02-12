try:
    from sglang._version import __version__, __version_tuple__
except ImportError:
    try:
        import importlib.metadata

        __version__ = importlib.metadata.version("sglang")
        __version_tuple__ = tuple(__version__.split("."))
    except Exception:
        try:
            import pathlib

            from setuptools_scm import get_version

            # point to the directory containing pyproject.toml.
            project_root = pathlib.Path(__file__).parent.parent.parent
            __version__ = get_version(
                root=str(project_root), fallback_version="1.0.1.dev0"
            )
            __version_tuple__ = tuple(__version__.split("."))
        except Exception:
            # Fallback for development without build
            __version__ = "1.0.1.dev0"
            __version_tuple__ = (1, 0, 1, "dev0")
