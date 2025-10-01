# Development Guide

## Development tips

- Use the Python virtualenv in `.venv`. If it doesn't exist, create one and install `kfp` and `kfp-kubernetes`.
- Always read components/README.md for available components to reuse and their documentation
- Favor reusing components
- Kubeflow Pipelines pipeline input parameters:
  - Do not use Union types
  - Add docstrings in the Google format
  - Optional inputs must have `= None` or a default value. Using the `Optional` type hint is not enough.
- Kubeflow Pipelines pipelines should always include a block to compile the pipeline. For example, to compile the pipeline called `yoda_finetune_and_evaluate`:

    ```python
    if __name__ == "__main__":
        from kfp import compiler

        compiler.Compiler().compile(
            pipeline_func=yoda_finetune_and_evaluate,
            package_path=__file__.replace(".py", ".yaml"),
        )
    ```

## Testing instructions

- Always run the pipeline file from the Python virtualenv to ensure it compiles correctly
