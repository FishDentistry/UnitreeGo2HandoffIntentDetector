1. Install all deps and this repo into env


## Installing the PoseFix Rule-Based Pipeline

The PoseFix rule-based comparative pipeline is included in the [PoseScript repository](https://github.com/naver/posescript). It does not require a pretrained model checkpoint.

> **Note:** PoseScript is research code and its original environment uses older versions of Python and PyTorch. The instructions below avoid replacing the PyTorch installation used by the existing inference server.

### 1. Clone the PoseScript Repository

Navigate to the directory where external repositories are stored and clone PoseScript:

```bash
git clone https://github.com/naver/posescript.git
cd posescript
```

The `main` branch contains the current PoseFix comparative pipeline:

```bash
git checkout main
```

The relevant implementation is located under:

```text
src/text2pose/posefix/
```

The rule-based correction-generation entry point is:

```python
from text2pose.posefix.correcting import main
```

### 2. Install PoseScript Without Replacing PyTorch

Activate the Python environment used by the existing server:

```bash
source /path/to/your/venv/bin/activate
```

On Windows:

```powershell
\path\to\your\venv\Scripts\Activate.ps1
```

From the root of the cloned `posescript` repository, install the package in editable mode without automatically installing its pinned dependencies:

```bash
pip install -e . --no-deps
```

Install the lightweight dependencies needed by the rule-based pipeline:

```bash
pip install roma networkx tabulate
```

Verify that Python can import the package:

```bash
python -c "import text2pose; print(text2pose.__file__)"
```

The printed path should point to the cloned PoseScript repository.

Next, test the PoseFix entry point:

```bash
python -c "from text2pose.posefix.correcting import main; print('PoseFix imported successfully')"
```

Do not initially run:

```bash
pip install -r requirements.txt
```

The repository requirements pin older versions of packages such as PyTorch. Installing them into the server environment may downgrade or replace the existing machine-learning dependencies.

### 3. Make the Optional Contact Dependency Lazy

PoseFix supports contact-based descriptions, but the hand-guidance integration does not use them. The pipeline will be called with:

```python
use_contact_codes=False
```

Some versions of PoseScript nevertheless import the optional contact module when `correcting.py` is first loaded. If the previous import test fails with an error involving `selfcontact`, `smplx`, or `format_contact_info`, modify the import so that it only runs when contact codes are enabled.

Open:

```text
posescript/src/text2pose/posefix/correcting.py
```

Find the top-level import resembling:

```python
from text2pose.posescript.format_contact_info import (
    from_joint_rotations_to_contact_list,
)
```

Remove or comment out that top-level import:

```python
# from text2pose.posescript.format_contact_info import (
#     from_joint_rotations_to_contact_list,
# )
```

In the same file, locate the block that handles contact codes:

```python
if use_contact_codes and joint_rotations is not None:
```

Import the contact function inside that conditional block:

```python
if use_contact_codes and joint_rotations is not None:
    from text2pose.posescript.format_contact_info import (
        from_joint_rotations_to_contact_list,
    )

    # Existing contact-processing code continues here.
```

This change does not modify PoseFix’s correction-generation method. It only prevents an unused optional dependency from being loaded when contact descriptions are disabled.

Run the import test again:

```bash
python -c "from text2pose.posefix.correcting import main; print('PoseFix imported successfully')"
```

A successful installation should print:

```text
PoseFix imported successfully
```
