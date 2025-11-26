# Scientific Named Entity Explorer

An interactive Streamlit application for mining scientific literature with pretrained named-entity-recognition (NER) models. Upload research abstracts from Excel spreadsheets, extract domain-specific entities (material, property, value, unit, method/process), and explore the results through rich filters, summaries, and downloadable reports.

## Prerequisites
- Python **3.10** or newer
- `git` (optional, for cloning)
- Internet connection for first-time model downloads from Hugging Face and Python package installation

> The models and `torch` wheels can be large (hundreds of MB). Ensure you have sufficient disk space and a reliable network connection.

## Quick Start
```bash
git clone https://github.com/k25063738/Named_entity_algorithm_project](https://github.com/Rkl2023/Named_entity_algorithm_project.git
cd Named_entity_algorithm_project
python3 install.py
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
streamlit run app.py
```

## Step-by-Step Installation
1. **Clone or download the project**
   ```bash
   git clone https://github.com/k25063738/Named_entity_algorithm_project
   cd Named_entity_algorithm_project
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate          # macOS/Linux
   # .venv\Scripts\activate           # Windows PowerShell
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit app**
   ```bash
   streamlit run app.py
   ```

> Prefer an automated setup? Run `python3 install.py` from the project root. It creates the virtual environment, installs dependencies, verifies Streamlit, and prints launch instructions.

## Using the App
1. Run `streamlit run app.py` (after activating your virtual environment).
2. Open the browser tab that Streamlit launches automatically (default: `http://localhost:8501`).
3. Select your preferred NER model and confidence threshold in the sidebar.
4. Upload an Excel file (`.xls` or `.xlsx`) containing at least **Title** and **Abstract** columns (other metadata like **DOI**, **Year** are optional).
5. Click **Run NER extraction**:
   - The app processes the abstracts with the selected pretrained model.
   - Extracted entities are tagged and displayed as colored chips.
   - Use the search box, entity checkboxes, and entity multiselect to filter papers.
6. Download filtered results as CSV, JSON, or Excel.

## Troubleshooting
- **`pip` cannot find a matching `torch` wheel**  
  Ensure you are on Python 3.10+ and upgrade `pip` (`pip install --upgrade pip`). For Apple Silicon or GPU builds, you may need platform-specific wheels from [PyTorch.org](https://pytorch.org/).

- **Model download is very slow**  
  The first run downloads large pretrained weights. Keep the terminal open until completion. Subsequent runs use the cached copy in `~/.cache/huggingface/`.

- **`streamlit` command not found after installation**  
  Activate the virtual environment (`source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\activate` on Windows) before running Streamlit.

- **Excel upload fails**  
  Confirm the file has the required columns and valid sheet data. The app reports missing columns or parsing errors directly in the UI.

- **Out-of-memory or very slow processing**  
  Large spreadsheets can be demanding. Increase the confidence threshold to reduce entity counts, or split very large files into smaller batches.

## Example Screenshots
_Add screenshots showcasing the upload interface, entity chips, and summary charts here._

---

Feel free to open issues or contribute improvements as you enhance the Scientific Named Entity Explorer for your research workflows.
