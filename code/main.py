# Imports
import pandas as pd
from dotenv import load_dotenv
import os,sys

# Import utils
from utils.functions import main

# Load environment variables from .env file
_ = load_dotenv()

# Extract path variables
cdc_tickets_path = os.getenv('CDC_TICKETS_PATH')
mcmn_tickets_path = os.getenv('MCMN_TICKETS_PATH')
system_message_path = os.getenv('SYSTEM_MESSAGE_PATH')
db_path = os.getenv('DB_PATH')

if not cdc_tickets_path or not mcmn_tickets_path or not system_message_path or not db_path:
    raise RuntimeError("There are missing environment variables. Check your .env and load_dotenv().")

# Load system message
SYSTEM_MESSAGE = open(system_message_path, 'r', encoding='utf-8').read()

# Main execution
if __name__ == '__main__':
    main(
        batchsize=int(sys.argv[1]) if len(sys.argv) > 1 else 100,
        minibatch_size=int(sys.argv[2]) if len(sys.argv) > 2 else 10,
        db_path=db_path,
        mcmn_tickets_path=mcmn_tickets_path,
        SYSTEM_MESSAGE=SYSTEM_MESSAGE
    )