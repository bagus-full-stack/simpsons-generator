"""Variables d'environnement communes à toute la suite de tests.

Doit s'exécuter avant le premier `import app...` : SKIP_MODEL_LOAD évite de
dépendre de torch/diffusers/un GPU pour tester le contrat de l'API.
"""

import os
import tempfile

os.environ.setdefault("SKIP_MODEL_LOAD", "1")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{tempfile.mktemp(suffix='.db')}")
os.environ.setdefault("API_KEYS", "")
os.environ.setdefault("ENABLE_NSFW_FILTER", "false")
