# Singularity: Deep Reasoning Model

### How to use uv:

uv is a modern python package manager & virtual environment manager

It is written in rust and performs 10x - 100x faster than pip + venv combo, which is already faster than conda

```uv init ``` is the command to initialize a new project in the current directory, this has already been done
``` uv sync ``` is a command that creates a .venv folder, tailored to the specifications in ```pyproject.toml```
``` uv.lock ``` is a file that contains versions of installed modules/packages. If you are familiar with npm, this is similar to ```package-lock.json```

``` uv run <filename>``` is a command that checks your pyproject.toml, ensures the venv is up to date with required packages, then executes python code
It is slower than raw python execution ```python 3 <filename>``` but it ensures there are no dependency issues

``` uvx pipreqs . --force ``` is a command to traverse your codebase and create a ```requirements.txt``` file with all python imports
``` uv add -r requirements.txt``` is the command to append or create the ```uv.lock``` file

## Developer pipeline:

0. ```git pull / git clone```
- optional: ```uv sync``, step 1 handles this automatically
1. ```uv run <filename>```, execute code without any dependency issues