# acados-ci image

Reusable CI image for `leap-c` and downstream repositories.

Published image:

```text
ghcr.io/leap-c/acados-ci:ubuntu24.04
ghcr.io/leap-c/acados-ci:latest
```

The image contains:

- Ubuntu 24.04 system build/runtime dependencies
- `uv`
- acados installed in `/opt/acados`
- `t_renderer` in `/opt/acados/bin`
- Python 3.11, 3.12, and 3.13 installed via `uv`

It intentionally does not install `leap-c`, `leap-c-lab`, or `mpc-sac`, so workflows can install the exact refs under test.

Environment variables:

```text
ACADOS_SOURCE_DIR=/opt/acados
ACADOS_INSTALL_DIR=/opt/acados
LD_LIBRARY_PATH=/opt/acados/lib
PATH=/opt/acados/bin:...
```

Downstream workflow usage:

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    container: ghcr.io/leap-c/acados-ci:ubuntu24.04
    strategy:
      matrix:
        python-version: ["3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        run: |
          uv venv --python ${{ matrix.python-version }} .venv
          echo "$PWD/.venv/bin" >> "$GITHUB_PATH"
      - name: Install acados Python interface
        run: uv pip install /opt/acados/interfaces/acados_template
```

After the first publish, ensure the GHCR package is public so downstream public repositories can pull it without credentials.
