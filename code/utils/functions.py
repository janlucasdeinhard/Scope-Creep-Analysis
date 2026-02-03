import pandas as pd
import json
from ollama import chat
import sqlite3

# Extract MCMn JSON to dataframe
def json_list_to_dataframe(json_list):
    rows = []
    for item in json_list:
        number = item.get("Number")
        resolution = item.get("resolution_note")
        message_blocks = []
        for msg in item.get("messages", []):
            timestamp = msg.get("timestamp") or "NO_TIMESTAMP"
            text = msg.get("text", "").strip()
            message_blocks.append(
                f"[{timestamp}]\n{text}"
            )
        timestamped_text_exchange = "\n\n---\n\n".join(message_blocks)
        rows.append({
            "number": number,
            "timestamped_text_exchange": timestamped_text_exchange,
            "resolution": resolution
        })
    return pd.DataFrame(rows)


# Load MCMn JSON from specified source and store as pandas frame
def load_mcmn_as_dataframe(cpath: str) -> pd.DataFrame:
    with open(cpath, 'r', encoding='utf-8') as f:
        mcmn_data = json.load(f)
    df = json_list_to_dataframe(mcmn_data)
    return df


# Call LLM on dataframe minibatch
def llm_process_minibatch(cdf: pd.DataFrame, system_message: str, verbose: bool = True) -> dict:
    # Make sure input data frame matches expected format
    expected_dtypes = {
        'number': 'object',
        'timestamped_text_exchange': 'object',
        'resolution': 'object'
    }
    if not all(cdf[col].dtype.name==expected_dtypes[col] for col in cdf.columns):
        raise Exception('Data type or column mismatch')
    # Dictionary to store results
    results = dict()
    # Iterate rows and examine each
    ctr = 1
    for _,cticket in cdf.iterrows():
        if verbose: print('{0}: Processing ticket {1}...'.format(ctr,cticket['number']), end='', flush=True)
        response = chat(
            model='qwen2.5:14b-instruct',
            messages=[
                {'role':'system','content':system_message},
                {'role':'user','content': f'''Ticket Number: {cticket['number']}
                                              Text Exchange: {cticket['timestamped_text_exchange']}
                                              Resolution Note: {cticket['resolution']}'''
            }],
            options={'temperature': 0.2},
        )
        results[cticket['number']] = response['message']['content']
        if verbose: print('done!\n')
        ctr += 1
    # Return results
    return results



# Generate minibatch from set of previously unprocessed tickets
def generate_batch(db_path:str,mcmn_tickets_path:str,batchsize:int=100) -> pd.DataFrame:
    # Load tickets
    df = load_mcmn_as_dataframe(mcmn_tickets_path)
    # Load existing tickets to exclude
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('select distinct number from inc where evident_root_cause is not null')
    known_incidents = [k[0] for k in cursor.fetchall()]
    conn.close()
    # Generate minibatch
    batch_df = df[~df['number'].isin(known_incidents)][:batchsize]
    return batch_df


# Main processing function
def main(batchsize,minibatch_size,db_path, mcmn_tickets_path,SYSTEM_MESSAGE):
    assert batchsize >= minibatch_size, 'Batch size must be greater than or equal to minibatch size. Note: Defaults if not specified in cmd is 100 and 10.'
    # Load batch of 100 unclassified tickets
    df = generate_batch(
        db_path=db_path,
        mcmn_tickets_path=mcmn_tickets_path,
        batchsize=batchsize
    )
    minibatch_size = minibatch_size
    # Iterate batch in minibatches
    ctr = 1 
    for i in range(0,df.shape[0],minibatch_size):
        msg = f'Processing batch {ctr}, records {i}:{i+minibatch_size}...'
        print(msg,end='',flush=True)
        # Create minibatch
        minibatch_df = df[i:i+minibatch_size].copy()
        # Process minibatch
        res = llm_process_minibatch(
            cdf=minibatch_df,
            system_message=SYSTEM_MESSAGE,
            verbose=False
        )
        # Attach results to minibatch dataframe
        minibatch_df.loc[:,'evident_root_cause'] = minibatch_df['number'].apply(lambda x: res[x])
        # Store to database
        with sqlite3.connect(db_path) as conn:
            minibatch_df.to_sql("inc", conn, if_exists="append", index=False)
        ctr += 1
        print('\r' + msg + ' Done!')
    print("All batches processed and stored to database.")
    return