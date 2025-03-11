from dotenv import load_dotenv
import os

# assume launch from directory above, found no other way, find_dotenv() broken
# when launching from repl at cwd=src/
# load_dotenv(os.getcwd()+"/prompters/.env")
load_dotenv(os.path.dirname(__file__)+f"/.env")
