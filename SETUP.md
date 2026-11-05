# Starting a new project from this template

Follow these steps for a new ML project using VS Code, GitHub and Google Colab on Windows.
Setting up a new project takes about 15 minutes, mostly waiting for PyTorch to download.

## Once only

Do these the first time. Later projects can skip them.

1. Install [Git](https://git-scm.com/downloads), Python 3.10 or newer,
   [VS Code](https://code.visualstudio.com/) and [GitHub Desktop](https://desktop.github.com/).
2. Create a GitHub token: **Settings > Developer settings > Personal access tokens >
   Fine-grained tokens > Generate new token**.
   - **Repository access:** Only select repositories, then pick any repo for now.
   - **Permissions:** Contents: Read-only.
   - Copy the token. GitHub only shows it once. Never paste it into a file or notebook.
3. In [Colab](https://colab.research.google.com/), open the key icon (**Secrets**), add a
   secret named `GITHUB_TOKEN`, paste the token and turn on **Notebook access**.
4. In Colab, open **File > Open notebook > GitHub**, tick **Include private repos** and
   allow access.

## For each new project

### 1. Create the repo

On the template's GitHub page, click **Use this template > Create a new repository**.
Choose a name such as `my-project` and set it to **Private**.

The repo name sets the project's names: `my-project` gets the Python package `my_project`
and the Drive folder `MyDrive/my-project/runs`.

### 2. Give Colab access

Edit your token on GitHub and add the new repo to its repository list.

### 3. Clone it

In GitHub Desktop, use **File > Clone repository**. Then open the folder in VS Code.

Keep the folder's full path under about 90 characters, such as
`C:\Users\you\Documents\GitHub\my-project`. PyTorch and Jupyter contain very long file
paths, and Windows limits their total length by default.

### 4. Set up the project

In a PowerShell terminal in the repo folder, run:

```powershell
python setup_project.py
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m pytest -q
```

`setup_project.py` renames the project throughout and deletes itself and this file.
All tests should pass.

### 5. Commit and push

In GitHub Desktop, commit everything with the message `Set up project`, then push.

### 6. Check Colab

1. In Colab, open `notebooks/01_gpu_check.ipynb` through **File > Open notebook > GitHub**.
2. Choose a GPU: **Runtime > Change runtime type > T4 GPU**.
3. Click **Runtime > Run all** and allow Google Drive access.

Check the output:

- `gpu` shows a GPU name, such as `Tesla T4`.
- `git_commit` matches your latest commit, and `git_dirty` is `False`.
- `runs dir` ends in `MyDrive/<your repo name>/runs`.
- `torch` has the same version number as `requirements.txt`. If it doesn't, follow
  "PyTorch version parity" in `README.md`.

### 7. Start your project

Everything now works from end to end. `notebooks/03_train_experiment.ipynb` trains the
example model on the sample data. Replace those in `src/<package>/ml/` as your project
takes shape. `README.md` covers the everyday workflow.

## If something goes wrong

| Problem | Likely cause | Fix |
|---|---|---|
| `Repository not found` or an authentication error in Colab | The token has expired or doesn't include this repo | Update the token on GitHub. If you made a new one, update the Colab secret too |
| `pip install` fails with `No such file or directory`, or `import torch` fails with a DLL error | The folder path is too long | Delete `.venv`, clone to a shorter path and repeat step 4 |
| `test_every_notebook_bootstraps_this_repo` fails | `setup_project.py` wasn't run | Run it, then run the tests again |
| Colab runs old code | The old code is still loaded | Push your changes, then **Runtime > Restart session** and run all cells |
