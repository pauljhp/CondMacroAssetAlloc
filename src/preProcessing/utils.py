import msal
from typing import (Union, Optional, List)
from office365.graph_client import GraphClient


class OneDriveAPI:
    def get_token(self,
        tenant_id: str,
        client_id: str,
        client_secret: str) -> str:
        """
        Acquire token via MSAL
        """
        authority_url = f'https://login.microsoftonline.com/{{{tenant_id}}}'
        app = msal.ConfidentialClientApplication(
            authority=authority_url,
            client_id=f'{client_id}',
            client_credential=f'{client_secret}'
        )
        token = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
        return token
    
    def __init__(self, 
            tenant_id: str,
            client_id: str,
            client_secret: str):
        self._tenant_id = tenant_id
        self._client_id = client_id
        self._client_secret = client_secret
        callback_func = self.get_token(self._tenant_id, self._client_id, self._client_secret)
        self.client = GraphClient(callback_func)
        self.drives = self.client.drives#.get().execute_query()

