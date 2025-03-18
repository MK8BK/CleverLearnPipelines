from dotenv import load_dotenv
import os
from pathlib import Path

p = os.path.join(str(Path(__file__).parent), ".env")
load_dotenv(p)
