# Release

The first distribution is prepared locally. It has not been published to PyPI.

From this directory:

```sh
python -m pip install -e '.[dev]'
pytest
ruff check .
ruff format --check .
mypy src
uv build
uvx twine check dist/*
```

Verify that `src/nullprotocol/py.typed`, license and package metadata are included in the wheel. Install the wheel in a clean environment before release. The workspace proof covers Python 3.10 and 3.14, the real API/gateway on an isolated database, scripted generation/embeddings and the runnable examples. It is not a model-quality benchmark.

Publishing is separate from building. Configure the owner's PyPI project credentials or trusted publishing, confirm the package name belongs to the project, then publish the exact tested artifacts. After successful publication, update the install instructions, public Docs, homepage SDK availability and cabinet connection examples together. Do not label `pip install nullprotocol` as released before the registry confirms it.
