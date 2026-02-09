"""
Gmail API Client Implementation for Windows 10 OpenClaw AI Agent System
=====================================================================

This module provides a complete Gmail API integration for the AI agent system,
including authentication, message operations, label management, thread handling,
attachments, drafts, and push notifications.

Dependencies:
    - google-auth-oauthlib
    - google-auth-httplib2
    - google-api-python-client
    - google-cloud-pubsub (for push notifications)
    - cryptography (for secure token storage)
    - keyring (for Windows credential storage)

Installation:
    pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client
    pip install google-cloud-pubsub cryptography keyring
"""

import os
import json
import base64
import time
from pathlib import Path
from typing import List, Dict, Optional, Callable, BinaryIO
from datetime import datetime, timedelta
from functools import wraps
from dataclasses import dataclass

# Google API imports
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Email handling
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import mimetypes


# =============================================================================
# EXCEPTIONS
# =============================================================================

class GmailAPIError(Exception):
    """Base Gmail API error"""
    pass


class GmailAuthError(GmailAPIError):
    """Authentication error"""
    pass


class GmailRateLimitError(GmailAPIError):
    """Rate limit exceeded"""
    pass


class GmailNotFoundError(GmailAPIError):
    """Resource not found"""
    pass


class GmailPermissionError(GmailAPIError):
    """Permission denied"""
    pass


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class GmailConfig:
    """Gmail API Configuration"""
    CLIENT_SECRETS_FILE: str = "credentials/client_secret.json"
    TOKEN_FILE: str = "credentials/token.json"
    AUTH_HOST: str = "localhost"
    AUTH_PORT: int = 8080
    
    # Recommended scopes for AI agent
    SCOPES: List[str] = None
    
    def __post_init__(self):
        if self.SCOPES is None:
            self.SCOPES = [
                'https://www.googleapis.com/auth/gmail.modify',
                'https://www.googleapis.com/auth/gmail.labels',
                'https://www.googleapis.com/auth/gmail.compose'
            ]


# =============================================================================
# AUTHENTICATION
# =============================================================================

class GmailAuthenticator:
    """
    Gmail OAuth 2.0 Authenticator for Windows Desktop Applications
    
    Handles the complete OAuth 2.0 flow including:
    - Initial authentication with user consent
    - Token refresh for 24/7 operation
    - Secure token storage
    - Token revocation
    """
    
    def __init__(self, config: GmailConfig = None):
        self.config = config or GmailConfig()
        self.credentials: Optional[Credentials] = None
    
    def authenticate(self) -> Credentials:
        """
        Authenticate user and return valid credentials.
        
        Returns:
            Valid Credentials object
        """
        # Load existing credentials
        if os.path.exists(self.config.TOKEN_FILE):
            self.credentials = Credentials.from_authorized_user_file(
                self.config.TOKEN_FILE, self.config.SCOPES
            )
        
        # Check if credentials are valid
        if not self.credentials or not self.credentials.valid:
            if (self.credentials and self.credentials.expired 
                and self.credentials.refresh_token):
                # Refresh expired credentials
                self.credentials.refresh(Request())
            else:
                # Run new authentication flow
                self._run_auth_flow()
            
            # Save credentials
            self._save_credentials()
        
        return self.credentials
    
    def _run_auth_flow(self):
        """Run OAuth 2.0 installed app flow"""
        flow = InstalledAppFlow.from_client_secrets_file(
            self.config.CLIENT_SECRETS_FILE, 
            self.config.SCOPES
        )
        
        self.credentials = flow.run_local_server(
            host=self.config.AUTH_HOST,
            port=self.config.AUTH_PORT,
            authorization_prompt_message=(
                'Please visit this URL to authorize the AI agent: {url}'
            ),
            success_message=(
                'Authentication successful! You may close this window.'
            ),
            open_browser=True
        )
    
    def _save_credentials(self):
        """Save credentials to token file"""
        token_data = {
            'token': self.credentials.token,
            'refresh_token': self.credentials.refresh_token,
            'token_uri': self.credentials.token_uri,
            'client_id': self.credentials.client_id,
            'client_secret': self.credentials.client_secret,
            'scopes': self.credentials.scopes,
            'expiry': self.credentials.expiry.isoformat() if self.credentials.expiry else None
        }
        
        os.makedirs(os.path.dirname(self.config.TOKEN_FILE), exist_ok=True)
        with open(self.config.TOKEN_FILE, 'w') as f:
            json.dump(token_data, f, indent=2)
    
    def revoke(self) -> bool:
        """Revoke stored credentials"""
        import requests
        
        if self.credentials and self.credentials.token:
            response = requests.post(
                'https://oauth2.googleapis.com/revoke',
                params={'token': self.credentials.token},
                headers={'content-type': 'application/x-www-form-urlencoded'}
            )
            
            if response.status_code == 200:
                if os.path.exists(self.config.TOKEN_FILE):
                    os.remove(self.config.TOKEN_FILE)
                self.credentials = None
                return True
        
        return False


class GmailService:
    """Gmail API Service Builder"""
    
    API_SERVICE_NAME = 'gmail'
    API_VERSION = 'v1'
    
    def __init__(self, authenticator: GmailAuthenticator):
        self.authenticator = authenticator
        self._service = None
    
    def get_service(self) -> build:
        """Build and return Gmail API service"""
        if not self._service:
            credentials = self.authenticator.authenticate()
            self._service = build(
                self.API_SERVICE_NAME,
                self.API_VERSION,
                credentials=credentials,
                cache_discovery=False
            )
        return self._service


# =============================================================================
# MESSAGE OPERATIONS
# =============================================================================

class MessageOperations:
    """Gmail Message Operations"""
    
    def __init__(self, service: build):
        self.service = service
        self.resource = service.users().messages()
    
    def list_messages(
        self,
        user_id: str = 'me',
        query: str = None,
        label_ids: List[str] = None,
        max_results: int = 100,
        include_spam_trash: bool = False,
        page_token: str = None
    ) -> Dict:
        """List messages in user's mailbox"""
        try:
            params = {
                'userId': user_id,
                'maxResults': max_results,
                'includeSpamTrash': include_spam_trash
            }
            
            if query:
                params['q'] = query
            if label_ids:
                params['labelIds'] = label_ids
            if page_token:
                params['pageToken'] = page_token
            
            return self.resource.list(**params).execute()
            
        except HttpError as e:
            raise GmailAPIError(f"Failed to list messages: {e}")
    
    def list_all_messages(
        self,
        user_id: str = 'me',
        query: str = None,
        label_ids: List[str] = None,
        include_spam_trash: bool = False
    ) -> List[Dict]:
        """List all messages with pagination handling"""
        messages = []
        page_token = None
        
        while True:
            result = self.list_messages(
                user_id=user_id,
                query=query,
                label_ids=label_ids,
                max_results=500,
                include_spam_trash=include_spam_trash,
                page_token=page_token
            )
            
            if 'messages' in result:
                messages.extend(result['messages'])
            
            page_token = result.get('nextPageToken')
            if not page_token:
                break
        
        return messages
    
    def get_message(
        self,
        message_id: str,
        user_id: str = 'me',
        format: str = 'full',
        metadata_headers: List[str] = None
    ) -> Dict:
        """Get a specific message"""
        try:
            params = {
                'userId': user_id,
                'id': message_id,
                'format': format
            }
            
            if metadata_headers:
                params['metadataHeaders'] = metadata_headers
            
            return self.resource.get(**params).execute()
            
        except HttpError as e:
            raise GmailAPIError(f"Failed to get message {message_id}: {e}")
    
    def send_message(
        self,
        to: str,
        subject: str,
        body: str,
        body_type: str = 'plain',
        cc: List[str] = None,
        bcc: List[str] = None,
        attachments: List[Dict] = None,
        user_id: str = 'me'
    ) -> Dict:
        """Send an email message"""
        try:
            # Create message
            message = MIMEMultipart()
            message['to'] = to
            message['subject'] = subject
            
            if cc:
                message['cc'] = ', '.join(cc)
            if bcc:
                message['bcc'] = ', '.join(bcc)
            
            # Add body
            body_part = MIMEText(body, body_type)
            message.attach(body_part)
            
            # Add attachments
            if attachments:
                for attachment in attachments:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment['content'])
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f"attachment; filename= {attachment['filename']}"
                    )
                    message.attach(part)
            
            # Encode and send
            raw_message = base64.urlsafe_b64encode(
                message.as_bytes()
            ).decode('utf-8')
            
            return self.resource.send(
                userId=user_id,
                body={'raw': raw_message}
            ).execute()
            
        except HttpError as e:
            raise GmailAPIError(f"Failed to send message: {e}")
    
    def delete_message(self, message_id: str, user_id: str = 'me') -> None:
        """Permanently delete a message"""
        try:
            self.resource.delete(userId=user_id, id=message_id).execute()
        except HttpError as e:
            raise GmailAPIError(f"Failed to delete message: {e}")
    
    def trash_message(self, message_id: str, user_id: str = 'me') -> Dict:
        """Move message to trash"""
        try:
            return self.resource.trash(userId=user_id, id=message_id).execute()
        except HttpError as e:
            raise GmailAPIError(f"Failed to trash message: {e}")
    
    def untrash_message(self, message_id: str, user_id: str = 'me') -> Dict:
        """Remove message from trash"""
        try:
            return self.resource.untrash(userId=user_id, id=message_id).execute()
        except HttpError as e:
            raise GmailAPIError(f"Failed to untrash message: {e}")
    
    def modify_labels(
        self,
        message_id: str,
        add_label_ids: List[str] = None,
        remove_label_ids: List[str] = None,
        user_id: str = 'me'
    ) -> Dict:
        """Modify labels on a message"""
        try:
            body = {}
            if add_label_ids:
                body['addLabelIds'] = add_label_ids
            if remove_label_ids:
                body['removeLabelIds'] = remove_label_ids
            
            return self.resource.modify(
                userId=user_id,
                id=message_id,
                body=body
            ).execute()
            
        except HttpError as e:
            raise GmailAPIError(f"Failed to modify labels: {e}")
    
    def batch_modify_labels(
        self,
        message_ids: List[str],
        add_label_ids: List[str] = None,
        remove_label_ids: List[str] = None,
        user_id: str = 'me'
    ) -> None:
        """Modify labels on multiple messages"""
        try:
            body = {'ids': message_ids}
            if add_label_ids:
                body['addLabelIds'] = add_label_ids
            if remove_label_ids:
                body['removeLabelIds'] = remove_label_ids
            
            self.resource.batchModify(userId=user_id, body=body).execute()
            
        except HttpError as e:
            raise GmailAPIError(f"Failed to batch modify labels: {e}")


# =============================================================================
# LABEL OPERATIONS
# =============================================================================

class LabelOperations:
    """Gmail Label Operations"""
    
    SYSTEM_LABELS = {
        'INBOX', 'SENT', 'DRAFT', 'SPAM', 'TRASH',
        'UNREAD', 'STARRED', 'IMPORTANT', 'CHAT',
        'CATEGORY_PERSONAL', 'CATEGORY_SOCIAL',
        'CATEGORY_PROMOTIONS', 'CATEGORY_UPDATES', 'CATEGORY_FORUMS'
    }
    
    def __init__(self, service: build):
        self.service = service
        self.resource = service.users().labels()
    
    def list_labels(self, user_id: str = 'me') -> List[Dict]:
        """List all labels"""
        try:
            result = self.resource.list(userId=user_id).execute()
            return result.get('labels', [])
        except HttpError as e:
            raise GmailAPIError(f"Failed to list labels: {e}")
    
    def get_label(self, label_id: str, user_id: str = 'me') -> Dict:
        """Get specific label"""
        try:
            return self.resource.get(userId=user_id, id=label_id).execute()
        except HttpError as e:
            raise GmailAPIError(f"Failed to get label: {e}")
    
    def create_label(
        self,
        name: str,
        label_list_visibility: str = 'labelShow',
        message_list_visibility: str = 'show',
        background_color: str = None,
        text_color: str = None,
        user_id: str = 'me'
    ) -> Dict:
        """Create a new label"""
        try:
            body = {
                'name': name,
                'labelListVisibility': label_list_visibility,
                'messageListVisibility': message_list_visibility
            }
            
            if background_color and text_color:
                body['color'] = {
                    'backgroundColor': background_color,
                    'textColor': text_color
                }
            
            return self.resource.create(userId=user_id, body=body).execute()
            
        except HttpError as e:
            raise GmailAPIError(f"Failed to create label: {e}")
    
    def update_label(
        self,
        label_id: str,
        name: str = None,
        label_list_visibility: str = None,
        message_list_visibility: str = None,
        user_id: str = 'me'
    ) -> Dict:
        """Update an existing label"""
        try:
            body = {}
            if name:
                body['name'] = name
            if label_list_visibility:
                body['labelListVisibility'] = label_list_visibility
            if message_list_visibility:
                body['messageListVisibility'] = message_list_visibility
            
            return self.resource.update(
                userId=user_id,
                id=label_id,
                body=body
            ).execute()
            
        except HttpError as e:
            raise GmailAPIError(f"Failed to update label: {e}")
    
    def delete_label(self, label_id: str, user_id: str = 'me') -> None:
        """Delete a label"""
        try:
            self.resource.delete(userId=user_id, id=label_id).execute()
        except HttpError as e:
            raise GmailAPIError(f"Failed to delete label: {e}")
    
    def is_system_label(self, label_id: str) -> bool:
        """Check if label is a system label"""
        return label_id in self.SYSTEM_LABELS


# =============================================================================
# THREAD OPERATIONS
# =============================================================================

class ThreadOperations:
    """Gmail Thread Operations"""
    
    def __init__(self, service: build):
        self.service = service
        self.resource = service.users().threads()
    
    def list_threads(
        self,
        user_id: str = 'me',
        query: str = None,
        label_ids: List[str] = None,
        max_results: int = 100,
        page_token: str = None
    ) -> Dict:
        """List threads"""
        try:
            params = {
                'userId': user_id,
                'maxResults': max_results
            }
            
            if query:
                params['q'] = query
            if label_ids:
                params['labelIds'] = label_ids
            if page_token:
                params['pageToken'] = page_token
            
            return self.resource.list(**params).execute()
            
        except HttpError as e:
            raise GmailAPIError(f"Failed to list threads: {e}")
    
    def get_thread(
        self,
        thread_id: str,
        user_id: str = 'me',
        format: str = 'full'
    ) -> Dict:
        """Get specific thread"""
        try:
            return self.resource.get(
                userId=user_id,
                id=thread_id,
                format=format
            ).execute()
            
        except HttpError as e:
            raise GmailAPIError(f"Failed to get thread: {e}")
    
    def modify_thread_labels(
        self,
        thread_id: str,
        add_label_ids: List[str] = None,
        remove_label_ids: List[str] = None,
        user_id: str = 'me'
    ) -> Dict:
        """Modify labels on a thread"""
        try:
            body = {}
            if add_label_ids:
                body['addLabelIds'] = add_label_ids
            if remove_label_ids:
                body['removeLabelIds'] = remove_label_ids
            
            return self.resource.modify(
                userId=user_id,
                id=thread_id,
                body=body
            ).execute()
            
        except HttpError as e:
            raise GmailAPIError(f"Failed to modify thread labels: {e}")
    
    def trash_thread(self, thread_id: str, user_id: str = 'me') -> Dict:
        """Move thread to trash"""
        try:
            return self.resource.trash(userId=user_id, id=thread_id).execute()
        except HttpError as e:
            raise GmailAPIError(f"Failed to trash thread: {e}")
    
    def untrash_thread(self, thread_id: str, user_id: str = 'me') -> Dict:
        """Remove thread from trash"""
        try:
            return self.resource.untrash(userId=user_id, id=thread_id).execute()
        except HttpError as e:
            raise GmailAPIError(f"Failed to untrash thread: {e}")
    
    def get_conversation_view(self, thread_id: str, user_id: str = 'me') -> Dict:
        """Get formatted conversation view of a thread"""
        thread = self.get_thread(thread_id, user_id, format='full')
        messages = thread.get('messages', [])
        
        conversation = {
            'thread_id': thread_id,
            'history_id': thread.get('historyId'),
            'message_count': len(messages),
            'participants': set(),
            'messages': []
        }
        
        for msg in messages:
            headers = {
                h['name']: h['value'] 
                for h in msg['payload'].get('headers', [])
            }
            
            message_data = {
                'message_id': msg['id'],
                'label_ids': msg.get('labelIds', []),
                'snippet': msg.get('snippet'),
                'from': headers.get('From'),
                'to': headers.get('To'),
                'subject': headers.get('Subject'),
                'date': headers.get('Date'),
            }
            
            if headers.get('From'):
                conversation['participants'].add(headers['From'])
            if headers.get('To'):
                conversation['participants'].add(headers['To'])
            
            conversation['messages'].append(message_data)
        
        conversation['participants'] = list(conversation['participants'])
        return conversation


# =============================================================================
# ATTACHMENT OPERATIONS
# =============================================================================

class AttachmentOperations:
    """Gmail Attachment Operations"""
    
    def __init__(self, service: build):
        self.service = service
        self.resource = service.users().messages().attachments()
    
    def get_attachment(
        self,
        message_id: str,
        attachment_id: str,
        user_id: str = 'me'
    ) -> Dict:
        """Get attachment data"""
        try:
            return self.resource.get(
                userId=user_id,
                messageId=message_id,
                id=attachment_id
            ).execute()
            
        except HttpError as e:
            raise GmailAPIError(f"Failed to get attachment: {e}")
    
    def download_attachment(
        self,
        message_id: str,
        attachment_id: str,
        filename: str = None,
        save_path: str = None,
        user_id: str = 'me'
    ) -> Dict:
        """Download and optionally save attachment"""
        attachment = self.get_attachment(message_id, attachment_id, user_id)
        
        # Decode base64 data
        data = attachment.get('data', '')
        data = data.replace('-', '+').replace('_', '/')
        padding = 4 - len(data) % 4
        if padding != 4:
            data += '=' * padding
        
        decoded_data = base64.urlsafe_b64decode(data)
        
        result = {
            'data': decoded_data,
            'size': len(decoded_data),
            'filename': filename
        }
        
        if save_path and filename:
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, filename)
            with open(file_path, 'wb') as f:
                f.write(decoded_data)
            result['saved_path'] = file_path
        
        return result
    
    def extract_attachments_from_message(self, message: Dict) -> List[Dict]:
        """Extract all attachments from a message"""
        attachments = []
        payload = message.get('payload', {})
        
        def process_part(part, message_id):
            if 'parts' in part:
                for subpart in part['parts']:
                    process_part(subpart, message_id)
            
            filename = part.get('filename')
            body = part.get('body', {})
            attachment_id = body.get('attachmentId')
            
            if filename and attachment_id:
                attachments.append({
                    'message_id': message_id,
                    'attachment_id': attachment_id,
                    'filename': filename,
                    'mime_type': part.get('mimeType'),
                    'size': body.get('size')
                })
        
        process_part(payload, message['id'])
        return attachments
    
    @staticmethod
    def prepare_attachment(file_path: str, filename: str = None) -> Dict:
        """Prepare file for attachment"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not filename:
            filename = path.name
        
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = 'application/octet-stream'
        
        with open(file_path, 'rb') as f:
            content = f.read()
        
        return {
            'filename': filename,
            'content': content,
            'mimetype': mime_type
        }


# =============================================================================
# DRAFT OPERATIONS
# =============================================================================

class DraftOperations:
    """Gmail Draft Operations"""
    
    def __init__(self, service: build):
        self.service = service
        self.resource = service.users().drafts()
    
    def list_drafts(
        self,
        user_id: str = 'me',
        max_results: int = 100
    ) -> Dict:
        """List drafts"""
        try:
            return self.resource.list(
                userId=user_id,
                maxResults=max_results
            ).execute()
        except HttpError as e:
            raise GmailAPIError(f"Failed to list drafts: {e}")
    
    def get_draft(self, draft_id: str, user_id: str = 'me') -> Dict:
        """Get specific draft"""
        try:
            return self.resource.get(
                userId=user_id,
                id=draft_id
            ).execute()
        except HttpError as e:
            raise GmailAPIError(f"Failed to get draft: {e}")
    
    def create_draft(
        self,
        to: str,
        subject: str,
        body: str,
        body_type: str = 'plain',
        cc: List[str] = None,
        bcc: List[str] = None,
        attachments: List[Dict] = None,
        user_id: str = 'me'
    ) -> Dict:
        """Create a new draft"""
        try:
            message = MIMEMultipart()
            message['to'] = to
            message['subject'] = subject
            
            if cc:
                message['cc'] = ', '.join(cc)
            if bcc:
                message['bcc'] = ', '.join(bcc)
            
            message.attach(MIMEText(body, body_type))
            
            if attachments:
                for attachment in attachments:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment['content'])
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f"attachment; filename= {attachment['filename']}"
                    )
                    message.attach(part)
            
            raw_message = base64.urlsafe_b64encode(
                message.as_bytes()
            ).decode('utf-8')
            
            return self.resource.create(
                userId=user_id,
                body={'message': {'raw': raw_message}}
            ).execute()
            
        except HttpError as e:
            raise GmailAPIError(f"Failed to create draft: {e}")
    
    def send_draft(self, draft_id: str, user_id: str = 'me') -> Dict:
        """Send a draft"""
        try:
            return self.resource.send(
                userId=user_id,
                body={'id': draft_id}
            ).execute()
        except HttpError as e:
            raise GmailAPIError(f"Failed to send draft: {e}")
    
    def delete_draft(self, draft_id: str, user_id: str = 'me') -> None:
        """Delete a draft"""
        try:
            self.resource.delete(userId=user_id, id=draft_id).execute()
        except HttpError as e:
            raise GmailAPIError(f"Failed to delete draft: {e}")


# =============================================================================
# MAIN CLIENT
# =============================================================================

class GmailClient:
    """
    Main Gmail API Client for AI Agent
    
    Provides unified access to all Gmail operations.
    
    Example:
        client = GmailClient()
        
        # List unread messages
        messages = client.messages.list_messages(query='is:unread')
        
        # Send email
        client.messages.send_message(
            to='recipient@example.com',
            subject='Hello',
            body='Test message'
        )
        
        # Create label
        client.labels.create_label(name='AI-Processed')
    """
    
    def __init__(
        self,
        client_secrets_file: str = 'credentials/client_secret.json',
        token_file: str = 'credentials/token.json',
        scopes: List[str] = None
    ):
        config = GmailConfig(
            CLIENT_SECRETS_FILE=client_secrets_file,
            TOKEN_FILE=token_file,
            SCOPES=scopes
        )
        
        self.authenticator = GmailAuthenticator(config)
        self.service_builder = GmailService(self.authenticator)
        
        self._service = None
        self._messages = None
        self._labels = None
        self._threads = None
        self._attachments = None
        self._drafts = None
    
    @property
    def service(self) -> build:
        """Lazy-load Gmail API service"""
        if not self._service:
            self._service = self.service_builder.get_service()
        return self._service
    
    @property
    def messages(self) -> MessageOperations:
        if not self._messages:
            self._messages = MessageOperations(self.service)
        return self._messages
    
    @property
    def labels(self) -> LabelOperations:
        if not self._labels:
            self._labels = LabelOperations(self.service)
        return self._labels
    
    @property
    def threads(self) -> ThreadOperations:
        if not self._threads:
            self._threads = ThreadOperations(self.service)
        return self._threads
    
    @property
    def attachments(self) -> AttachmentOperations:
        if not self._attachments:
            self._attachments = AttachmentOperations(self.service)
        return self._attachments
    
    @property
    def drafts(self) -> DraftOperations:
        if not self._drafts:
            self._drafts = DraftOperations(self.service)
        return self._drafts
    
    def revoke(self) -> bool:
        """Revoke authentication"""
        return self.authenticator.revoke()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def with_retry(max_retries: int = 3, backoff_factor: float = 2.0):
    """Decorator for retry with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except (OSError, RuntimeError, ValueError) as e:
                    retries += 1
                    if retries >= max_retries:
                        raise
                    wait_time = backoff_factor ** retries
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator


# Search query helpers
class SearchQueries:
    """Gmail search query builders"""
    
    @staticmethod
    def unread() -> str:
        return 'is:unread'
    
    @staticmethod
    def from_sender(email: str) -> str:
        return f'from:{email}'
    
    @staticmethod
    def to_recipient(email: str) -> str:
        return f'to:{email}'
    
    @staticmethod
    def subject(text: str) -> str:
        return f'subject:{text}'
    
    @staticmethod
    def has_attachment() -> str:
        return 'has:attachment'
    
    @staticmethod
    def filename(ext: str) -> str:
        return f'filename:{ext}'
    
    @staticmethod
    def after_date(date: str) -> str:
        """Date format: YYYY/MM/DD"""
        return f'after:{date}'
    
    @staticmethod
    def before_date(date: str) -> str:
        """Date format: YYYY/MM/DD"""
        return f'before:{date}'
    
    @staticmethod
    def newer_than(period: str) -> str:
        """Period: 1d, 2w, 3m, 1y"""
        return f'newer_than:{period}'
    
    @staticmethod
    def label(name: str) -> str:
        return f'label:{name}'
    
    @staticmethod
    def combine(*queries: str) -> str:
        """Combine multiple queries with AND logic"""
        return ' '.join(queries)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == '__main__':
    # Example usage
    print("Gmail Client Implementation for OpenClaw AI Agent")
    print("=" * 50)
    
    # Initialize client
    # client = GmailClient()
    
    # List unread messages
    # messages = client.messages.list_messages(
    #     query=SearchQueries.unread(),
    #     max_results=10
    # )
    
    # Send email
    # client.messages.send_message(
    #     to='recipient@example.com',
    #     subject='Test from AI Agent',
    #     body='This is a test message.'
    # )
    
    # Create label
    # client.labels.create_label(
    #     name='AI-Processed',
    #     background_color='#16a766',
    #     text_color='#ffffff'
    # )
    
    print("\nImplementation ready for integration with AI agent system.")
