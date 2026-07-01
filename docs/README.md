To build the documentation from this folder locally, run:

```bash
make html
```

Then open `build/html/index.html` in your favorite browser.

## Live preview

For a live-reloading preview that rebuilds on every save, run:

```bash
make livehtml
```

Then open the printed URL (default <http://127.0.0.1:8000>).

Both targets require the docs dependencies:

```bash
pip install -e ".[docs]"
```
