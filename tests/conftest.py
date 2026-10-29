"""Variables d'environnement communes à toute la suite de tests.

Doit s'exécuter avant le premier `import app...` : SKIP_MODEL_LOAD évite de
dépendre de torch/diffusers/un GPU pour tester le contrat de l'API.
"""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("SKIP_MODEL_LOAD", "1")
_tmp = Path(tempfile.mkdtemp())
os.environ.setdefault("DATABASE_URL", f"sqlite:///{(_tmp / 'app.db').as_posix()}")
os.environ.setdefault("OUTPUT_FOLDER", str(_tmp))
os.environ.setdefault("API_KEYS", "")
os.environ.setdefault("ENABLE_NSFW_FILTER", "false")
