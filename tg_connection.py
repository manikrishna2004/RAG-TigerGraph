import os
import requests
from dotenv import load_dotenv
import pyTigerGraph as tg

load_dotenv()

import os
import requests
import pyTigerGraph as tg

import os
import requests
import pyTigerGraph as tg
from dotenv import load_dotenv
 
load_dotenv()

def get_tg_connection():
    host = os.environ["TG_HOST"]
    graph = os.environ["TG_GRAPH"]
    secret = os.environ["TG_SECRET"]

    resp = requests.post(f"{host}/gsql/v1/tokens", json={"secret": secret, "lifetime": 999999}, timeout=60)
    if resp.status_code != 200:
        print(f"\n[TigerGraph Auth Error] Status: {resp.status_code}")
        print(f"Raw response: {resp.text}\n")
        resp.raise_for_status()

    token = resp.json()["token"]

    conn = tg.TigerGraphConnection(host=host, graphname=graph, apiToken=token)
    
    # Set custom RESTPP header using pyTigerGraph's extraHeaders dictionary
    conn.extraHeaders = {"GSQL-TIMEOUT": "60000"}
    
    return conn