# Contribution Guide

## Install by editable mode

```bash
make install-editable
```

## Install development requirements

```bash
make dev-env
```

## Lint

```bash
make check-lint
```

## Auto format

```bash
make auto-format
```

## Build docs

```bash
make doc
```

## Preview docs

```bash
cd build/docs_build/html
python3 -m http.server {PORT}
# open browser: http://localhost:{PORT}
```

## Run test

```bash
make test
```

## Acknowledgement

Our work is inspired by many existing deep learning algorithm frameworks, such as [OpenMM Lab](https://github.com/openmm), [Hugging face transformers](https://github.com/huggingface/transformers) [Hugging face diffusers](https://github.com/huggingface/diffusers) etc. We would like to thank all the contributors of all the open-source projects that we use in RoboOrchard.
