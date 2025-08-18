## Build
### Clone and checkout the repository
Clone the repo and checkout the master branch.

### Create a Virtual Environment
Whichever build system you choose, make sure to create a venv first.
So that dependencies are installed for this project and not in your global Python environment.
```shell
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
```

### Install Dependencies
You can install the dependencies using either `uv` or `pip`.  
Again, make sure you are in the virtual environment.

#### Build with uv
```bash
uv sync
```

#### Or build it with pip
```bash
pip install -e
```


## Running the application
Run the application (will open a browser window):
```shell
uv run streamlit run .\gpt-batcher\app.py
```

## Providing an API Key
Currently only the OpenAI API is supported.  
You need to provide an API key to use the application.  
You can create one here: https://platform.openai.com/api-keys

Then paste it in the sidebar on the left:
![api_key.png](docs/api_key.png)

The key is not stored anywhere and you will have to re-enter it
every time you start the application or refresh the page.