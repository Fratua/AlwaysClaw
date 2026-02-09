# Gmail API Integration Architecture
## Technical Specification for Windows 10 OpenClaw AI Agent System

**Version:** 1.0  
**Date:** 2025-01-20  
**Target Platform:** Windows 10  
**API Version:** Gmail API v1  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Gmail API v1 Endpoint Coverage](#2-gmail-api-v1-endpoint-coverage)
3. [OAuth 2.0 Authentication Flow](#3-oauth-20-authentication-flow)
4. [Message Operations](#4-message-operations)
5. [Label Management](#5-label-management)
6. [Thread Operations](#6-thread-operations)
7. [Attachment Handling](#7-attachment-handling)
8. [Draft Management](#8-draft-management)
9. [Push Notifications via Webhook](#9-push-notifications-via-webhook)
10. [Implementation Architecture](#10-implementation-architecture)
11. [Error Handling & Rate Limiting](#11-error-handling--rate-limiting)
12. [Security Considerations](#12-security-considerations)

---

## 1. Executive Summary

This document provides a comprehensive technical specification for integrating Gmail API v1 into a Windows 10-based OpenClaw-inspired AI agent system. The architecture supports full email lifecycle management including authentication, message operations, label management, thread handling, attachments, drafts, and real-time push notifications.

### Key Features
- **Full OAuth 2.0 Integration** with offline access and automatic token refresh
- **Complete CRUD operations** for messages, labels, threads, and drafts
- **Real-time push notifications** via Google Cloud Pub/Sub
- **Attachment handling** with Base64 encoding/decoding
- **Thread-based conversation view** for contextual email management
- **Robust error handling** with exponential backoff

---

## 2. Gmail API v1 Endpoint Coverage

### 2.1 REST Resources Overview

| Resource | Description | Key Methods |
|----------|-------------|-------------|
| `v1.users` | User profile and watch management | `getProfile`, `stop`, `watch` |
| `v1.users.drafts` | Draft message management | `create`, `delete`, `get`, `list`, `send`, `update` |
| `v1.users.history` | Mailbox change history | `list` |
| `v1.users.labels` | Label management | `create`, `delete`, `get`, `list`, `patch`, `update` |
| `v1.users.messages` | Message operations | `batchDelete`, `batchModify`, `delete`, `get`, `import`, `insert`, `list`, `modify`, `send`, `trash`, `untrash` |
| `v1.users.messages.attachments` | Attachment handling | `get` |
| `v1.users.settings` | User settings | Various settings operations |
| `v1.users.threads` | Thread operations | `delete`, `get`, `list`, `modify`, `trash`, `untrash` |

### 2.2 Base Endpoint

```
https://gmail.googleapis.com
```

### 2.3 Discovery Document

```
https://gmail.googleapis.com/$discovery/rest?version=v1
```

---

## 3. OAuth 2.0 Authentication Flow

### 3.1 Required Scopes

| Scope | Permission Level | Use Case |
|-------|-----------------|----------|
| `https://www.googleapis.com/auth/gmail.readonly` | Read-only | Read messages, threads, labels |
| `https://www.googleapis.com/auth/gmail.send` | Send only | Send emails only |
| `https://www.googleapis.com/auth/gmail.labels` | Labels management | Create, update, delete labels |
| `https://www.googleapis.com/auth/gmail.insert` | Insert/Import | Insert messages into mailbox |
| `https://www.googleapis.com/auth/gmail.compose` | Compose | Create, read, update, delete drafts |
| `https://www.googleapis.com/auth/gmail.modify` | Read/Write (recommended) | All operations except permanent delete |
| `https://mail.google.com/` | Full access | All read/write operations (use with caution) |

**Recommended Scope for AI Agent:**
```python
SCOPES = [
    'https://www.googleapis.com/auth/gmail.modify',
    'https://www.googleapis.com/auth/gmail.labels',
    'https://www.googleapis.com/auth/gmail.compose'
]
```

### 3.2 Authentication Flow Implementation

#### Step 1: Client Secrets Configuration

```python
# config/gmail_auth.py
import os
from pathlib import Path

class GmailAuthConfig:
    """Gmail OAuth 2.0 Configuration"""
    
    CLIENT_SECRETS_FILE = Path("credentials/client_secret.json")
    TOKEN_FILE = Path("credentials/token.json")
    
    SCOPES = [
        'https://www.googleapis.com/auth/gmail.modify',
        'https://www.googleapis.com/auth/gmail.labels',
        'https://www.googleapis.com/auth/gmail.compose'
    ]
    
    REDIRECT_URI = "http://localhost:8080/oauth2callback"
    
    # For installed app flow (Windows desktop)
    AUTH_HOST = "localhost"
    AUTH_PORT = 8080
```

#### Step 2: Installed App Flow (Windows Desktop)

```python
# auth/gmail_oauth.py
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import json
import os
from pathlib import Path

class GmailAuthenticator:
    """Gmail OAuth 2.0 Authenticator for Windows Desktop"""
    
    def __init__(self, client_secrets_file: str, token_file: str, scopes: list):
        self.client_secrets_file = client_secrets_file
        self.token_file = token_file
        self.scopes = scopes
        self.credentials = None
    
    def authenticate(self) -> Credentials:
        """Authenticate user and return credentials"""
        
        # Load existing credentials if available
        if os.path.exists(self.token_file):
            self.credentials = Credentials.from_authorized_user_file(
                self.token_file, self.scopes
            )
        
        # If no valid credentials, run authentication flow
        if not self.credentials or not self.credentials.valid:
            if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                # Refresh expired credentials
                self.credentials.refresh(Request())
            else:
                # Run new authentication flow
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.client_secrets_file, self.scopes
                )
                self.credentials = flow.run_local_server(
                    host='localhost',
                    port=8080,
                    authorization_prompt_message='Please visit this URL to authorize the AI agent: {url}',
                    success_message='Authentication successful! You may close this window.',
                    open_browser=True
                )
            
            # Save credentials for future runs
            self._save_credentials()
        
        return self.credentials
    
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
        
        os.makedirs(os.path.dirname(self.token_file), exist_ok=True)
        with open(self.token_file, 'w') as f:
            json.dump(token_data, f)
    
    def revoke_credentials(self) -> bool:
        """Revoke stored credentials"""
        import requests
        
        if self.credentials and self.credentials.token:
            revoke = requests.post(
                'https://oauth2.googleapis.com/revoke',
                params={'token': self.credentials.token},
                headers={'content-type': 'application/x-www-form-urlencoded'}
            )
            
            if revoke.status_code == 200:
                if os.path.exists(self.token_file):
                    os.remove(self.token_file)
                self.credentials = None
                return True
        
        return False
```

#### Step 3: Service Builder

```python
# auth/gmail_service.py
from googleapiclient.discovery import build
from .gmail_oauth import GmailAuthenticator

class GmailService:
    """Gmail API Service Builder"""
    
    API_SERVICE_NAME = 'gmail'
    API_VERSION = 'v1'
    
    def __init__(self, authenticator: GmailAuthenticator):
        self.authenticator = authenticator
        self.service = None
    
    def get_service(self):
        """Build and return Gmail API service"""
        if not self.service:
            credentials = self.authenticator.authenticate()
            self.service = build(
                self.API_SERVICE_NAME, 
                self.API_VERSION, 
                credentials=credentials,
                cache_discovery=False
            )
        return self.service
```

---

## 4. Message Operations

### 4.1 Message Resource Methods

| Method | HTTP | Description |
|--------|------|-------------|
| `list` | GET | List messages in mailbox |
| `get` | GET | Get specific message |
| `send` | POST | Send a message |
| `delete` | DELETE | Permanently delete message |
| `trash` | POST | Move message to trash |
| `untrash` | POST | Remove message from trash |
| `modify` | POST | Modify message labels |
| `batchDelete` | POST | Delete multiple messages |
| `batchModify` | POST | Modify labels on multiple messages |
| `import` | POST | Import message (SMTP-like) |
| `insert` | POST | Insert message (IMAP-like) |

### 4.2 Message Operations Implementation

```python
# operations/messages.py
from googleapiclient.errors import HttpError
from base64 import urlsafe_b64encode, urlsafe_b64decode
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
from typing import List, Dict, Optional, BinaryIO

class MessageOperations:
    """Gmail Message Operations"""
    
    def __init__(self, service):
        self.service = service
        self.messages_resource = service.users().messages()
    
    # ==================== LIST MESSAGES ====================
    
    def list_messages(
        self, 
        user_id: str = 'me',
        query: str = None,
        label_ids: List[str] = None,
        max_results: int = 100,
        include_spam_trash: bool = False,
        page_token: str = None
    ) -> Dict:
        """
        List messages in user's mailbox.
        
        Args:
            user_id: User's email address or 'me'
            query: Gmail search query (e.g., 'from:sender@example.com')
            label_ids: Filter by label IDs
            max_results: Maximum results per page (1-500)
            include_spam_trash: Include spam/trash
            page_token: Token for pagination
        
        Returns:
            Dict with 'messages', 'nextPageToken', 'resultSizeEstimate'
        """
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
            
            result = self.messages_resource.list(**params).execute()
            return result
            
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
    
    # ==================== GET MESSAGE ====================
    
    def get_message(
        self,
        message_id: str,
        user_id: str = 'me',
        format: str = 'full',
        metadata_headers: List[str] = None
    ) -> Dict:
        """
        Get a specific message.
        
        Args:
            message_id: Message ID
            user_id: User's email address or 'me'
            format: 'minimal', 'full', 'raw', 'metadata'
            metadata_headers: Specific headers to retrieve (for 'metadata' format)
        
        Returns:
            Message object with payload, headers, etc.
        """
        try:
            params = {
                'userId': user_id,
                'id': message_id,
                'format': format
            }
            
            if metadata_headers:
                params['metadataHeaders'] = metadata_headers
            
            return self.messages_resource.get(**params).execute()
            
        except HttpError as e:
            raise GmailAPIError(f"Failed to get message {message_id}: {e}")
    
    # ==================== SEND MESSAGE ====================
    
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
        """
        Send an email message.
        
        Args:
            to: Recipient email address
            subject: Email subject
            body: Email body content
            body_type: 'plain' or 'html'
            cc: CC recipients
            bcc: BCC recipients
            attachments: List of attachment dicts {'filename': str, 'content': bytes, 'mimetype': str}
            user_id: User's email address or 'me'
        
        Returns:
            Sent message metadata
        """
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
            raw_message = urlsafe_b64encode(message.as_bytes()).decode('utf-8')
            body = {'raw': raw_message}
            
            sent_message = self.messages_resource.send(
                userId=user_id,
                body=body
            ).execute()
            
            return sent_message
            
        except HttpError as e:
            raise GmailAPIError(f"Failed to send message: {e}")
    
    # ==================== DELETE/TRASH ====================
    
    def delete_message(self, message_id: str, user_id: str = 'me') -> None:
        """Permanently delete a message (cannot be undone)"""
        try:
            self.messages_resource.delete(userId=user_id, id=message_id).execute()
        except HttpError as e:
            raise GmailAPIError(f"Failed to delete message {message_id}: {e}")
    
    def trash_message(self, message_id: str, user_id: str = 'me') -> Dict:
        """Move message to trash"""
        try:
            return self.messages_resource.trash(userId=user_id, id=message_id).execute()
        except HttpError as e:
            raise GmailAPIError(f"Failed to trash message {message_id}: {e}")
    
    def untrash_message(self, message_id: str, user_id: str = 'me') -> Dict:
        """Remove message from trash"""
        try:
            return self.messages_resource.untrash(userId=user_id, id=message_id).execute()
        except HttpError as e:
            raise GmailAPIError(f"Failed to untrash message {message_id}: {e}")
    
    # ==================== MODIFY LABELS ====================
    
    def modify_labels(
        self,
        message_id: str,
        add_label_ids: List[str] = None,
        remove_label_ids: List[str] = None,
        user_id: str = 'me'
    ) -> Dict:
        """
        Modify labels on a message.
        
        Args:
            message_id: Message ID
            add_label_ids: Label IDs to add
            remove_label_ids: Label IDs to remove
            user_id: User's email address or 'me'
        """
        try:
            body = {}
            if add_label_ids:
                body['addLabelIds'] = add_label_ids
            if remove_label_ids:
                body['removeLabelIds'] = remove_label_ids
            
            return self.messages_resource.modify(
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
            
            self.messages_resource.batchModify(userId=user_id, body=body).execute()
            
        except HttpError as e:
            raise GmailAPIError(f"Failed to batch modify labels: {e}")
    
    def batch_delete_messages(self, message_ids: List[str], user_id: str = 'me') -> None:
        """Delete multiple messages"""
        try:
            body = {'ids': message_ids}
            self.messages_resource.batchDelete(userId=user_id, body=body).execute()
        except HttpError as e:
            raise GmailAPIError(f"Failed to batch delete messages: {e}")
```

### 4.3 Search Query Syntax

```python
# Common Gmail search queries for AI agent
SEARCH_QUERIES = {
    'unread': 'is:unread',
    'read': 'is:read',
    'starred': 'is:starred',
    'important': 'is:important',
    'in_inbox': 'in:inbox',
    'in_sent': 'in:sent',
    'in_drafts': 'in:drafts',
    'in_trash': 'in:trash',
    'in_spam': 'in:spam',
    'from_sender': lambda sender: f'from:{sender}',
    'to_recipient': lambda recipient: f'to:{recipient}',
    'subject_contains': lambda text: f'subject:{text}',
    'has_attachment': 'has:attachment',
    'attachment_type': lambda ext: f'filename:{ext}',
    'larger_than': lambda size: f'larger:{size}',
    'smaller_than': lambda size: f'smaller:{size}',
    'after_date': lambda date: f'after:{date}',  # YYYY/MM/DD
    'before_date': lambda date: f'before:{date}',
    'on_date': lambda date: f'on:{date}',
    'newer_than': lambda period: f'newer_than:{period}',  # e.g., '1d', '2w', '3m'
    'older_than': lambda period: f'older_than:{period}',
    'label': lambda label: f'label:{label}',
    'category': lambda cat: f'category:{cat}',  # primary, social, promotions, updates, forums
    'combined': lambda **kwargs: ' '.join(f'{k}:{v}' for k, v in kwargs.items())
}
```

---

## 5. Label Management

### 5.1 Label Resource Methods

| Method | HTTP | Description |
|--------|------|-------------|
| `list` | GET | List all labels |
| `get` | GET | Get specific label |
| `create` | POST | Create new label |
| `update` | PUT | Update label |
| `patch` | PATCH | Partial update label |
| `delete` | DELETE | Delete label |

### 5.2 System Labels

| Label ID | Name | Description |
|----------|------|-------------|
| `INBOX` | Inbox | Main inbox |
| `SENT` | Sent | Sent messages |
| `DRAFT` | Drafts | Draft messages |
| `SPAM` | Spam | Spam messages |
| `TRASH` | Trash | Deleted messages |
| `UNREAD` | Unread | Unread messages |
| `STARRED` | Starred | Starred messages |
| `IMPORTANT` | Important | Important messages |
| `CHAT` | Chat | Chat messages |
| `CATEGORY_PERSONAL` | Personal | Personal category |
| `CATEGORY_SOCIAL` | Social | Social category |
| `CATEGORY_PROMOTIONS` | Promotions | Promotions category |
| `CATEGORY_UPDATES` | Updates | Updates category |
| `CATEGORY_FORUMS` | Forums | Forums category |

### 5.3 Label Management Implementation

```python
# operations/labels.py
from googleapiclient.errors import HttpError
from typing import List, Dict, Optional

class LabelOperations:
    """Gmail Label Operations"""
    
    # System label IDs
    SYSTEM_LABELS = {
        'INBOX', 'SENT', 'DRAFT', 'SPAM', 'TRASH',
        'UNREAD', 'STARRED', 'IMPORTANT', 'CHAT',
        'CATEGORY_PERSONAL', 'CATEGORY_SOCIAL',
        'CATEGORY_PROMOTIONS', 'CATEGORY_UPDATES', 'CATEGORY_FORUMS'
    }
    
    def __init__(self, service):
        self.service = service
        self.labels_resource = service.users().labels()
    
    # ==================== LIST LABELS ====================
    
    def list_labels(self, user_id: str = 'me') -> List[Dict]:
        """
        List all labels for user.
        
        Returns:
            List of label objects with id, name, type, etc.
        """
        try:
            result = self.labels_resource.list(userId=user_id).execute()
            return result.get('labels', [])
        except HttpError as e:
            raise GmailAPIError(f"Failed to list labels: {e}")
    
    def get_label(self, label_id: str, user_id: str = 'me') -> Dict:
        """Get specific label details"""
        try:
            return self.labels_resource.get(userId=user_id, id=label_id).execute()
        except HttpError as e:
            raise GmailAPIError(f"Failed to get label {label_id}: {e}")
    
    def get_label_by_name(self, label_name: str, user_id: str = 'me') -> Optional[Dict]:
        """Find label by name"""
        labels = self.list_labels(user_id)
        for label in labels:
            if label['name'].lower() == label_name.lower():
                return label
        return None
    
    # ==================== CREATE LABEL ====================
    
    def create_label(
        self,
        name: str,
        label_list_visibility: str = 'labelShow',
        message_list_visibility: str = 'show',
        background_color: str = None,
        text_color: str = None,
        user_id: str = 'me'
    ) -> Dict:
        """
        Create a new label.
        
        Args:
            name: Label name (max 225 characters)
            label_list_visibility: 'labelShow', 'labelShowIfUnread', 'labelHide'
            message_list_visibility: 'show', 'hide'
            background_color: Hex color code (e.g., '#000000')
            text_color: Hex color code (e.g., '#ffffff')
            user_id: User's email address or 'me'
        
        Returns:
            Created label object
        """
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
            
            return self.labels_resource.create(userId=user_id, body=body).execute()
            
        except HttpError as e:
            raise GmailAPIError(f"Failed to create label: {e}")
    
    # ==================== UPDATE LABEL ====================
    
    def update_label(
        self,
        label_id: str,
        name: str = None,
        label_list_visibility: str = None,
        message_list_visibility: str = None,
        background_color: str = None,
        text_color: str = None,
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
            if background_color and text_color:
                body['color'] = {
                    'backgroundColor': background_color,
                    'textColor': text_color
                }
            
            return self.labels_resource.update(
                userId=user_id,
                id=label_id,
                body=body
            ).execute()
            
        except HttpError as e:
            raise GmailAPIError(f"Failed to update label {label_id}: {e}")
    
    def patch_label(
        self,
        label_id: str,
        name: str = None,
        label_list_visibility: str = None,
        message_list_visibility: str = None,
        user_id: str = 'me'
    ) -> Dict:
        """Partially update a label"""
        try:
            body = {}
            
            if name:
                body['name'] = name
            if label_list_visibility:
                body['labelListVisibility'] = label_list_visibility
            if message_list_visibility:
                body['messageListVisibility'] = message_list_visibility
            
            return self.labels_resource.patch(
                userId=user_id,
                id=label_id,
                body=body
            ).execute()
            
        except HttpError as e:
            raise GmailAPIError(f"Failed to patch label {label_id}: {e}")
    
    # ==================== DELETE LABEL ====================
    
    def delete_label(self, label_id: str, user_id: str = 'me') -> None:
        """Delete a label"""
        try:
            self.labels_resource.delete(userId=user_id, id=label_id).execute()
        except HttpError as e:
            raise GmailAPIError(f"Failed to delete label {label_id}: {e}")
    
    # ==================== UTILITY METHODS ====================
    
    def is_system_label(self, label_id: str) -> bool:
        """Check if label is a system label"""
        return label_id in self.SYSTEM_LABELS
    
    def get_or_create_label(self, name: str, user_id: str = 'me') -> Dict:
        """Get existing label or create new one"""
        existing = self.get_label_by_name(name, user_id)
        if existing:
            return existing
        return self.create_label(name, user_id=user_id)
    
    def get_label_colors(self) -> List[Dict]:
        """Get available label colors"""
        return [
            {'background': '#000000', 'text': '#ffffff'},
            {'background': '#434343', 'text': '#ffffff'},
            {'background': '#666666', 'text': '#ffffff'},
            {'background': '#999999', 'text': '#ffffff'},
            {'background': '#cccccc', 'text': '#000000'},
            {'background': '#efefef', 'text': '#000000'},
            {'background': '#f3f3f3', 'text': '#000000'},
            {'background': '#ffffff', 'text': '#000000'},
            {'background': '#fb4c2f', 'text': '#ffffff'},
            {'background': '#ffad47', 'text': '#000000'},
            {'background': '#fad165', 'text': '#000000'},
            {'background': '#16a766', 'text': '#ffffff'},
            {'background': '#43d692', 'text': '#000000'},
            {'background': '#4a86e8', 'text': '#ffffff'},
            {'background': '#a479e2', 'text': '#ffffff'},
            {'background': '#f691b3', 'text': '#000000'},
            {'background': '#b0120a', 'text': '#ffffff'},
            {'background': '#b36d00', 'text': '#ffffff'},
            {'background': '#7a4706', 'text': '#ffffff'},
            {'background': '#0e5e2f', 'text': '#ffffff'},
            {'background': '#005229', 'text': '#ffffff'},
            {'background': '#0057e7', 'text': '#ffffff'},
            {'background': '#20124d', 'text': '#ffffff'},
            {'background': '#762583', 'text': '#ffffff'},
        ]
```

---

## 6. Thread Operations

### 6.1 Thread Resource Methods

| Method | HTTP | Description |
|--------|------|-------------|
| `list` | GET | List threads |
| `get` | GET | Get specific thread |
| `modify` | POST | Modify thread labels |
| `delete` | DELETE | Delete thread |
| `trash` | POST | Move thread to trash |
| `untrash` | POST | Remove thread from trash |

### 6.2 Thread Operations Implementation

```python
# operations/threads.py
from googleapiclient.errors import HttpError
from typing import List, Dict, Optional

class ThreadOperations:
    """Gmail Thread Operations"""
    
    def __init__(self, service):
        self.service = service
        self.threads_resource = service.users().threads()
    
    # ==================== LIST THREADS ====================
    
    def list_threads(
        self,
        user_id: str = 'me',
        query: str = None,
        label_ids: List[str] = None,
        max_results: int = 100,
        include_spam_trash: bool = False,
        page_token: str = None
    ) -> Dict:
        """
        List threads in user's mailbox.
        
        Args:
            user_id: User's email address or 'me'
            query: Gmail search query
            label_ids: Filter by label IDs
            max_results: Maximum results per page
            include_spam_trash: Include spam/trash
            page_token: Token for pagination
        
        Returns:
            Dict with 'threads', 'nextPageToken', 'resultSizeEstimate'
        """
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
            
            return self.threads_resource.list(**params).execute()
            
        except HttpError as e:
            raise GmailAPIError(f"Failed to list threads: {e}")
    
    def list_all_threads(
        self,
        user_id: str = 'me',
        query: str = None,
        label_ids: List[str] = None,
        include_spam_trash: bool = False
    ) -> List[Dict]:
        """List all threads with pagination"""
        threads = []
        page_token = None
        
        while True:
            result = self.list_threads(
                user_id=user_id,
                query=query,
                label_ids=label_ids,
                max_results=500,
                include_spam_trash=include_spam_trash,
                page_token=page_token
            )
            
            if 'threads' in result:
                threads.extend(result['threads'])
            
            page_token = result.get('nextPageToken')
            if not page_token:
                break
        
        return threads
    
    # ==================== GET THREAD ====================
    
    def get_thread(
        self,
        thread_id: str,
        user_id: str = 'me',
        format: str = 'full',
        metadata_headers: List[str] = None
    ) -> Dict:
        """
        Get a specific thread with all messages.
        
        Args:
            thread_id: Thread ID
            user_id: User's email address or 'me'
            format: 'minimal', 'full', 'metadata'
            metadata_headers: Specific headers for metadata format
        
        Returns:
            Thread object with messages array
        """
        try:
            params = {
                'userId': user_id,
                'id': thread_id,
                'format': format
            }
            
            if metadata_headers:
                params['metadataHeaders'] = metadata_headers
            
            return self.threads_resource.get(**params).execute()
            
        except HttpError as e:
            raise GmailAPIError(f"Failed to get thread {thread_id}: {e}")
    
    def get_thread_messages(self, thread_id: str, user_id: str = 'me') -> List[Dict]:
        """Get all messages in a thread"""
        thread = self.get_thread(thread_id, user_id)
        return thread.get('messages', [])
    
    # ==================== MODIFY THREAD ====================
    
    def modify_thread_labels(
        self,
        thread_id: str,
        add_label_ids: List[str] = None,
        remove_label_ids: List[str] = None,
        user_id: str = 'me'
    ) -> Dict:
        """
        Modify labels on all messages in a thread.
        
        Args:
            thread_id: Thread ID
            add_label_ids: Label IDs to add
            remove_label_ids: Label IDs to remove
            user_id: User's email address or 'me'
        """
        try:
            body = {}
            if add_label_ids:
                body['addLabelIds'] = add_label_ids
            if remove_label_ids:
                body['removeLabelIds'] = remove_label_ids
            
            return self.threads_resource.modify(
                userId=user_id,
                id=thread_id,
                body=body
            ).execute()
            
        except HttpError as e:
            raise GmailAPIError(f"Failed to modify thread labels: {e}")
    
    # ==================== DELETE/TRASH ====================
    
    def delete_thread(self, thread_id: str, user_id: str = 'me') -> None:
        """Permanently delete a thread"""
        try:
            self.threads_resource.delete(userId=user_id, id=thread_id).execute()
        except HttpError as e:
            raise GmailAPIError(f"Failed to delete thread {thread_id}: {e}")
    
    def trash_thread(self, thread_id: str, user_id: str = 'me') -> Dict:
        """Move thread to trash"""
        try:
            return self.threads_resource.trash(userId=user_id, id=thread_id).execute()
        except HttpError as e:
            raise GmailAPIError(f"Failed to trash thread {thread_id}: {e}")
    
    def untrash_thread(self, thread_id: str, user_id: str = 'me') -> Dict:
        """Remove thread from trash"""
        try:
            return self.threads_resource.untrash(userId=user_id, id=thread_id).execute()
        except HttpError as e:
            raise GmailAPIError(f"Failed to untrash thread {thread_id}: {e}")
    
    # ==================== CONVERSATION VIEW ====================
    
    def get_conversation_view(
        self,
        thread_id: str,
        user_id: str = 'me'
    ) -> Dict:
        """
        Get formatted conversation view of a thread.
        
        Returns:
            Dict with thread metadata and formatted messages
        """
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
            headers = {h['name']: h['value'] for h in msg['payload'].get('headers', [])}
            
            message_data = {
                'message_id': msg['id'],
                'thread_id': msg['threadId'],
                'label_ids': msg.get('labelIds', []),
                'snippet': msg.get('snippet'),
                'from': headers.get('From'),
                'to': headers.get('To'),
                'subject': headers.get('Subject'),
                'date': headers.get('Date'),
                'internal_date': msg.get('internalDate'),
                'history_id': msg.get('historyId'),
                'size_estimate': msg.get('sizeEstimate')
            }
            
            # Extract participants
            if headers.get('From'):
                conversation['participants'].add(headers['From'])
            if headers.get('To'):
                conversation['participants'].add(headers['To'])
            
            conversation['messages'].append(message_data)
        
        conversation['participants'] = list(conversation['participants'])
        return conversation
```

---

## 7. Attachment Handling

### 7.1 Attachment Resource Methods

| Method | HTTP | Description |
|--------|------|-------------|
| `get` | GET | Get attachment content |

### 7.2 Attachment Handling Implementation

```python
# operations/attachments.py
from googleapiclient.errors import HttpError
from base64 import urlsafe_b64decode, urlsafe_b64encode
import os
import mimetypes
from typing import Dict, Optional, BinaryIO, Tuple
from pathlib import Path

class AttachmentOperations:
    """Gmail Attachment Operations"""
    
    def __init__(self, service):
        self.service = service
        self.attachments_resource = service.users().messages().attachments()
    
    # ==================== DOWNLOAD ATTACHMENT ====================
    
    def get_attachment(
        self,
        message_id: str,
        attachment_id: str,
        user_id: str = 'me'
    ) -> Dict:
        """
        Get attachment metadata and data.
        
        Args:
            message_id: Message ID containing attachment
            attachment_id: Attachment ID
            user_id: User's email address or 'me'
        
        Returns:
            Dict with 'data', 'size', 'attachmentId'
        """
        try:
            return self.attachments_resource.get(
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
        """
        Download and optionally save attachment.
        
        Args:
            message_id: Message ID
            attachment_id: Attachment ID
            filename: Optional filename for saving
            save_path: Directory to save file
            user_id: User's email address or 'me'
        
        Returns:
            Dict with 'data' (bytes), 'filename', 'size', 'saved_path'
        """
        # Get attachment data
        attachment = self.get_attachment(message_id, attachment_id, user_id)
        
        # Decode base64 data (URL-safe base64)
        data = attachment.get('data', '')
        # Replace URL-safe characters
        data = data.replace('-', '+').replace('_', '/')
        # Add padding if needed
        padding = 4 - len(data) % 4
        if padding != 4:
            data += '=' * padding
        
        decoded_data = urlsafe_b64decode(data)
        
        result = {
            'data': decoded_data,
            'size': len(decoded_data),
            'attachment_id': attachment_id,
            'filename': filename
        }
        
        # Save to file if path provided
        if save_path and filename:
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, filename)
            with open(file_path, 'wb') as f:
                f.write(decoded_data)
            result['saved_path'] = file_path
        
        return result
    
    # ==================== EXTRACT FROM MESSAGE ====================
    
    def extract_attachments_from_message(
        self,
        message: Dict,
        save_path: str = None
    ) -> List[Dict]:
        """
        Extract all attachments from a message.
        
        Args:
            message: Message object from get_message()
            save_path: Optional directory to save attachments
        
        Returns:
            List of attachment info dicts
        """
        attachments = []
        payload = message.get('payload', {})
        
        def process_part(part, message_id):
            """Recursively process message parts"""
            if 'parts' in part:
                for subpart in part['parts']:
                    process_part(subpart, message_id)
            
            # Check if part is an attachment
            filename = part.get('filename')
            body = part.get('body', {})
            attachment_id = body.get('attachmentId')
            
            if filename and attachment_id:
                attachment_info = {
                    'message_id': message_id,
                    'attachment_id': attachment_id,
                    'filename': filename,
                    'mime_type': part.get('mimeType'),
                    'size': body.get('size')
                }
                attachments.append(attachment_info)
        
        process_part(payload, message['id'])
        return attachments
    
    def download_all_attachments(
        self,
        message_id: str,
        save_path: str,
        user_id: str = 'me'
    ) -> List[Dict]:
        """
        Download all attachments from a message.
        
        Args:
            message_id: Message ID
            save_path: Directory to save attachments
            user_id: User's email address or 'me'
        
        Returns:
            List of downloaded attachment info
        """
        from .messages import MessageOperations
        
        msg_ops = MessageOperations(self.service)
        message = msg_ops.get_message(message_id, user_id, format='full')
        
        attachment_infos = self.extract_attachments_from_message(message)
        downloaded = []
        
        for info in attachment_infos:
            result = self.download_attachment(
                message_id=info['message_id'],
                attachment_id=info['attachment_id'],
                filename=info['filename'],
                save_path=save_path,
                user_id=user_id
            )
            downloaded.append(result)
        
        return downloaded
    
    # ==================== PREPARE FOR SENDING ====================
    
    @staticmethod
    def prepare_attachment_for_sending(
        file_path: str,
        filename: str = None,
        mime_type: str = None
    ) -> Dict:
        """
        Prepare file for attachment to email.
        
        Args:
            file_path: Path to file
            filename: Optional custom filename
            mime_type: Optional MIME type (auto-detected if not provided)
        
        Returns:
            Dict with 'filename', 'content' (bytes), 'mimetype'
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine filename
        if not filename:
            filename = path.name
        
        # Determine MIME type
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                mime_type = 'application/octet-stream'
        
        # Read file content
        with open(file_path, 'rb') as f:
            content = f.read()
        
        return {
            'filename': filename,
            'content': content,
            'mimetype': mime_type
        }
    
    @staticmethod
    def prepare_attachment_from_bytes(
        content: bytes,
        filename: str,
        mime_type: str = 'application/octet-stream'
    ) -> Dict:
        """Prepare bytes for attachment"""
        return {
            'filename': filename,
            'content': content,
            'mimetype': mime_type
        }
```

---

## 8. Draft Management

### 8.1 Draft Resource Methods

| Method | HTTP | Description |
|--------|------|-------------|
| `list` | GET | List drafts |
| `get` | GET | Get specific draft |
| `create` | POST | Create new draft |
| `update` | PUT | Update existing draft |
| `delete` | DELETE | Delete draft |
| `send` | POST | Send draft |

### 8.2 Draft Management Implementation

```python
# operations/drafts.py
from googleapiclient.errors import HttpError
from base64 import urlsafe_b64encode
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Dict, Optional

class DraftOperations:
    """Gmail Draft Operations"""
    
    def __init__(self, service):
        self.service = service
        self.drafts_resource = service.users().drafts()
    
    # ==================== LIST DRAFTS ====================
    
    def list_drafts(
        self,
        user_id: str = 'me',
        max_results: int = 100,
        page_token: str = None
    ) -> Dict:
        """
        List drafts in user's mailbox.
        
        Args:
            user_id: User's email address or 'me'
            max_results: Maximum results per page
            page_token: Token for pagination
        
        Returns:
            Dict with 'drafts', 'nextPageToken', 'resultSizeEstimate'
        """
        try:
            params = {
                'userId': user_id,
                'maxResults': max_results
            }
            
            if page_token:
                params['pageToken'] = page_token
            
            return self.drafts_resource.list(**params).execute()
            
        except HttpError as e:
            raise GmailAPIError(f"Failed to list drafts: {e}")
    
    def list_all_drafts(self, user_id: str = 'me') -> List[Dict]:
        """List all drafts with pagination"""
        drafts = []
        page_token = None
        
        while True:
            result = self.list_drafts(
                user_id=user_id,
                max_results=500,
                page_token=page_token
            )
            
            if 'drafts' in result:
                drafts.extend(result['drafts'])
            
            page_token = result.get('nextPageToken')
            if not page_token:
                break
        
        return drafts
    
    # ==================== GET DRAFT ====================
    
    def get_draft(
        self,
        draft_id: str,
        user_id: str = 'me',
        format: str = 'full'
    ) -> Dict:
        """
        Get a specific draft.
        
        Args:
            draft_id: Draft ID
            user_id: User's email address or 'me'
            format: 'minimal', 'full', 'raw'
        
        Returns:
            Draft object with message
        """
        try:
            return self.drafts_resource.get(
                userId=user_id,
                id=draft_id,
                format=format
            ).execute()
            
        except HttpError as e:
            raise GmailAPIError(f"Failed to get draft {draft_id}: {e}")
    
    # ==================== CREATE DRAFT ====================
    
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
        """
        Create a new draft.
        
        Args:
            to: Recipient email address
            subject: Email subject
            body: Email body
            body_type: 'plain' or 'html'
            cc: CC recipients
            bcc: BCC recipients
            attachments: List of attachment dicts
            user_id: User's email address or 'me'
        
        Returns:
            Created draft object
        """
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
            
            # Encode message
            raw_message = urlsafe_b64encode(message.as_bytes()).decode('utf-8')
            
            body = {
                'message': {
                    'raw': raw_message
                }
            }
            
            return self.drafts_resource.create(
                userId=user_id,
                body=body
            ).execute()
            
        except HttpError as e:
            raise GmailAPIError(f"Failed to create draft: {e}")
    
    def create_draft_from_message(
        self,
        message: MIMEMultipart,
        user_id: str = 'me'
    ) -> Dict:
        """Create draft from existing MIMEMultipart message"""
        try:
            raw_message = urlsafe_b64encode(message.as_bytes()).decode('utf-8')
            body = {'message': {'raw': raw_message}}
            
            return self.drafts_resource.create(
                userId=user_id,
                body=body
            ).execute()
            
        except HttpError as e:
            raise GmailAPIError(f"Failed to create draft: {e}")
    
    # ==================== UPDATE DRAFT ====================
    
    def update_draft(
        self,
        draft_id: str,
        to: str = None,
        subject: str = None,
        body: str = None,
        body_type: str = 'plain',
        cc: List[str] = None,
        bcc: List[str] = None,
        attachments: List[Dict] = None,
        user_id: str = 'me'
    ) -> Dict:
        """Update an existing draft"""
        try:
            # Get existing draft
            existing = self.get_draft(draft_id, user_id)
            existing_message = existing.get('message', {})
            
            # Extract existing headers
            headers = {h['name']: h['value'] for h in existing_message.get('payload', {}).get('headers', [])}
            
            # Create updated message
            message = MIMEMultipart()
            message['to'] = to or headers.get('To', '')
            message['subject'] = subject or headers.get('Subject', '')
            
            if cc:
                message['cc'] = ', '.join(cc)
            elif headers.get('Cc'):
                message['cc'] = headers['Cc']
            
            if bcc:
                message['bcc'] = ', '.join(bcc)
            elif headers.get('Bcc'):
                message['bcc'] = headers['Bcc']
            
            # Add body
            body_content = body or existing_message.get('snippet', '')
            body_part = MIMEText(body_content, body_type)
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
            
            # Encode and update
            raw_message = urlsafe_b64encode(message.as_bytes()).decode('utf-8')
            body = {
                'message': {
                    'raw': raw_message
                }
            }
            
            return self.drafts_resource.update(
                userId=user_id,
                id=draft_id,
                body=body
            ).execute()
            
        except HttpError as e:
            raise GmailAPIError(f"Failed to update draft {draft_id}: {e}")
    
    # ==================== DELETE DRAFT ====================
    
    def delete_draft(self, draft_id: str, user_id: str = 'me') -> None:
        """Delete a draft"""
        try:
            self.drafts_resource.delete(userId=user_id, id=draft_id).execute()
        except HttpError as e:
            raise GmailAPIError(f"Failed to delete draft {draft_id}: {e}")
    
    # ==================== SEND DRAFT ====================
    
    def send_draft(self, draft_id: str, user_id: str = 'me') -> Dict:
        """
        Send an existing draft.
        
        Args:
            draft_id: Draft ID to send
            user_id: User's email address or 'me'
        
        Returns:
            Sent message object
        """
        try:
            body = {'id': draft_id}
            return self.drafts_resource.send(
                userId=user_id,
                body=body
            ).execute()
            
        except HttpError as e:
            raise GmailAPIError(f"Failed to send draft {draft_id}: {e}")
```

---

## 9. Push Notifications via Webhook

### 9.1 Push Notification Architecture

```
Gmail Account → Gmail Push Notification → Google Cloud Pub/Sub → Webhook → AI Agent
```

### 9.2 Prerequisites

1. Google Cloud Project with Pub/Sub API enabled
2. Gmail API Push notification access
3. Webhook endpoint (HTTPS)
4. Service account or OAuth credentials

### 9.3 Push Notification Implementation

```python
# notifications/push_notifications.py
from googleapiclient.errors import HttpError
from google.cloud import pubsub_v1
import json
from typing import Dict, Callable
from datetime import datetime, timedelta

class PushNotificationManager:
    """Gmail Push Notification Manager"""
    
    # Gmail API Push service account
    GMAIL_PUSH_SERVICE_ACCOUNT = 'gmail-api-push@system.gserviceaccount.com'
    
    def __init__(self, service, project_id: str = None):
        self.service = service
        self.project_id = project_id
        self.pubsub_client = None
        if project_id:
            self.pubsub_client = pubsub_v1.PublisherClient()
    
    # ==================== WATCH / START NOTIFICATIONS ====================
    
    def watch_mailbox(
        self,
        topic_name: str,
        label_ids: list = None,
        label_filter_action: str = 'include',
        user_id: str = 'me'
    ) -> Dict:
        """
        Set up watch on mailbox for push notifications.
        
        Args:
            topic_name: Full Pub/Sub topic name (projects/PROJECT_ID/topics/TOPIC_NAME)
            label_ids: List of label IDs to filter (e.g., ['INBOX'])
            label_filter_action: 'include' or 'exclude'
            user_id: User's email address or 'me'
        
        Returns:
            Watch response with 'historyId', 'expiration'
        """
        try:
            body = {
                'topicName': topic_name,
                'labelFilterAction': label_filter_action
            }
            
            if label_ids:
                body['labelIds'] = label_ids
            
            return self.service.users().watch(
                userId=user_id,
                body=body
            ).execute()
            
        except HttpError as e:
            raise GmailAPIError(f"Failed to set up watch: {e}")
    
    def stop_watch(self, user_id: str = 'me') -> None:
        """Stop receiving push notifications"""
        try:
            self.service.users().stop(userId=user_id).execute()
        except HttpError as e:
            raise GmailAPIError(f"Failed to stop watch: {e}")
    
    # ==================== PUB/SUB SETUP ====================
    
    def create_pubsub_topic(self, topic_id: str) -> str:
        """
        Create Pub/Sub topic for Gmail notifications.
        
        Args:
            topic_id: Topic ID (not full name)
        
        Returns:
            Full topic name
        """
        if not self.pubsub_client:
            raise ValueError("Pub/Sub client not initialized. Provide project_id.")
        
        topic_path = self.pubsub_client.topic_path(self.project_id, topic_id)
        
        try:
            topic = self.pubsub_client.create_topic(request={"name": topic_path})
            return topic.name
        except Exception as e:
            # Topic may already exist
            return topic_path
    
    def grant_publish_permission(self, topic_name: str) -> None:
        """
        Grant Gmail service account permission to publish to topic.
        
        Args:
            topic_name: Full topic name
        """
        from google.iam.v1 import policy_pb2
        
        if not self.pubsub_client:
            raise ValueError("Pub/Sub client not initialized.")
        
        # Get current IAM policy
        policy = self.pubsub_client.get_iam_policy(request={"resource": topic_name})
        
        # Add Gmail push service account as publisher
        binding = policy_pb2.Binding(
            role="roles/pubsub.publisher",
            members=[f"serviceAccount:{self.GMAIL_PUSH_SERVICE_ACCOUNT}"]
        )
        policy.bindings.append(binding)
        
        # Update policy
        self.pubsub_client.set_iam_policy(
            request={"resource": topic_name, "policy": policy}
        )
    
    def create_push_subscription(
        self,
        topic_name: str,
        subscription_id: str,
        push_endpoint: str
    ) -> str:
        """
        Create push subscription for webhook notifications.
        
        Args:
            topic_name: Full topic name
            subscription_id: Subscription ID
            push_endpoint: HTTPS webhook URL
        
        Returns:
            Full subscription name
        """
        if not self.pubsub_client:
            raise ValueError("Pub/Sub client not initialized.")
        
        subscriber = pubsub_v1.SubscriberClient()
        subscription_path = subscriber.subscription_path(
            self.project_id, subscription_id
        )
        
        push_config = pubsub_v1.types.PushConfig(
            push_endpoint=push_endpoint
        )
        
        try:
            subscription = subscriber.create_subscription(
                request={
                    "name": subscription_path,
                    "topic": topic_name,
                    "push_config": push_config,
                    "ack_deadline_seconds": 60
                }
            )
            return subscription.name
        except Exception as e:
            # Subscription may already exist
            return subscription_path
    
    # ==================== WEBHOOK HANDLER ====================
    
    @staticmethod
    def parse_notification(payload: bytes) -> Dict:
        """
        Parse incoming Pub/Sub notification.
        
        Args:
            payload: Raw request body
        
        Returns:
            Parsed notification data
        """
        from base64 import b64decode
        
        data = json.loads(payload)
        message_data = data.get('message', {})
        
        # Decode base64 data
        encoded_data = message_data.get('data', '')
        decoded_data = b64decode(encoded_data).decode('utf-8')
        notification = json.loads(decoded_data)
        
        return {
            'email_address': notification.get('emailAddress'),
            'history_id': notification.get('historyId'),
            'publish_time': message_data.get('publishTime'),
            'message_id': message_data.get('messageId'),
            'subscription': data.get('subscription')
        }
    
    def process_notification(
        self,
        notification: Dict,
        message_handler: Callable = None
    ) -> None:
        """
        Process incoming notification.
        
        Args:
            notification: Parsed notification dict
            message_handler: Callback function for new messages
        """
        email_address = notification.get('email_address')
        history_id = notification.get('history_id')
        
        # Get history to find changes
        history = self.service.users().history().list(
            userId='me',
            startHistoryId=history_id
        ).execute()
        
        history_records = history.get('history', [])
        
        for record in history_records:
            # Process message additions
            if 'messagesAdded' in record:
                for msg_added in record['messagesAdded']:
                    message = msg_added.get('message', {})
                    if message_handler:
                        message_handler(message)
    
    # ==================== WATCH RENEWAL ====================
    
    def get_watch_expiration(self, watch_response: Dict) -> datetime:
        """Get expiration datetime from watch response"""
        expiration_ms = int(watch_response.get('expiration', 0))
        return datetime.fromtimestamp(expiration_ms / 1000)
    
    def should_renew_watch(self, watch_response: Dict, buffer_hours: int = 24) -> bool:
        """
        Check if watch should be renewed.
        
        Args:
            watch_response: Watch response from watch_mailbox()
            buffer_hours: Hours before expiration to renew
        
        Returns:
            True if renewal needed
        """
        expiration = self.get_watch_expiration(watch_response)
        renewal_threshold = datetime.now() + timedelta(hours=buffer_hours)
        return expiration <= renewal_threshold
```

### 9.4 Webhook Endpoint Example (Flask)

```python
# notifications/webhook_server.py
from flask import Flask, request, jsonify
from .push_notifications import PushNotificationManager

app = Flask(__name__)

@app.route('/webhook/gmail', methods=['POST'])
def gmail_webhook():
    """Handle Gmail push notifications"""
    
    # Verify request (implement signature verification)
    # Google Cloud Pub/Sub doesn't sign requests by default
    # Consider using Cloud Tasks or verifying via other means
    
    # Parse notification
    payload = request.get_data()
    notification = PushNotificationManager.parse_notification(payload)
    
    # Process notification
    # - Get new messages
    # - Trigger AI agent processing
    # - Send notifications
    
    email_address = notification['email_address']
    history_id = notification['history_id']
    
    # Acknowledge message
    return jsonify({'status': 'acknowledged'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, ssl_context='adhoc')
```

---

## 10. Implementation Architecture

### 10.1 Project Structure

```
gmail_integration/
├── __init__.py
├── config/
│   ├── __init__.py
│   └── gmail_auth.py          # Authentication configuration
├── auth/
│   ├── __init__.py
│   ├── gmail_oauth.py         # OAuth 2.0 flow
│   └── gmail_service.py       # Service builder
├── operations/
│   ├── __init__.py
│   ├── messages.py            # Message operations
│   ├── labels.py              # Label management
│   ├── threads.py             # Thread operations
│   ├── attachments.py         # Attachment handling
│   └── drafts.py              # Draft management
├── notifications/
│   ├── __init__.py
│   ├── push_notifications.py  # Push notification setup
│   └── webhook_server.py      # Webhook endpoint
├── utils/
│   ├── __init__.py
│   ├── exceptions.py          # Custom exceptions
│   ├── parsers.py             # Message parsers
│   └── validators.py          # Input validators
├── models/
│   ├── __init__.py
│   ├── message.py             # Message data model
│   ├── thread.py              # Thread data model
│   └── label.py               # Label data model
└── tests/
    └── test_operations.py     # Unit tests
```

### 10.2 Main Client Class

```python
# gmail_client.py
from auth.gmail_service import GmailService
from auth.gmail_oauth import GmailAuthenticator
from operations.messages import MessageOperations
from operations.labels import LabelOperations
from operations.threads import ThreadOperations
from operations.attachments import AttachmentOperations
from operations.drafts import DraftOperations
from notifications.push_notifications import PushNotificationManager

class GmailClient:
    """Main Gmail API Client for AI Agent"""
    
    def __init__(
        self,
        client_secrets_file: str = 'credentials/client_secret.json',
        token_file: str = 'credentials/token.json',
        scopes: list = None,
        project_id: str = None
    ):
        # Initialize authenticator
        self.authenticator = GmailAuthenticator(
            client_secrets_file=client_secrets_file,
            token_file=token_file,
            scopes=scopes or [
                'https://www.googleapis.com/auth/gmail.modify',
                'https://www.googleapis.com/auth/gmail.labels',
                'https://www.googleapis.com/auth/gmail.compose'
            ]
        )
        
        # Initialize service
        self.service_builder = GmailService(self.authenticator)
        self._service = None
        
        # Initialize operation handlers
        self._messages = None
        self._labels = None
        self._threads = None
        self._attachments = None
        self._drafts = None
        self._notifications = None
        
        self.project_id = project_id
    
    @property
    def service(self):
        """Lazy-load Gmail API service"""
        if not self._service:
            self._service = self.service_builder.get_service()
        return self._service
    
    @property
    def messages(self) -> MessageOperations:
        """Message operations"""
        if not self._messages:
            self._messages = MessageOperations(self.service)
        return self._messages
    
    @property
    def labels(self) -> LabelOperations:
        """Label operations"""
        if not self._labels:
            self._labels = LabelOperations(self.service)
        return self._labels
    
    @property
    def threads(self) -> ThreadOperations:
        """Thread operations"""
        if not self._threads:
            self._threads = ThreadOperations(self.service)
        return self._threads
    
    @property
    def attachments(self) -> AttachmentOperations:
        """Attachment operations"""
        if not self._attachments:
            self._attachments = AttachmentOperations(self.service)
        return self._attachments
    
    @property
    def drafts(self) -> DraftOperations:
        """Draft operations"""
        if not self._drafts:
            self._drafts = DraftOperations(self.service)
        return self._drafts
    
    @property
    def notifications(self) -> PushNotificationManager:
        """Push notification manager"""
        if not self._notifications:
            self._notifications = PushNotificationManager(
                self.service,
                self.project_id
            )
        return self._notifications
    
    def revoke(self) -> bool:
        """Revoke authentication"""
        return self.authenticator.revoke_credentials()
```

---

## 11. Error Handling & Rate Limiting

### 11.1 Custom Exceptions

```python
# utils/exceptions.py

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
```

### 11.2 Rate Limiting

```python
# utils/rate_limiter.py
import time
from functools import wraps

class RateLimiter:
    """Gmail API Rate Limiter"""
    
    # Gmail API quota limits
    QUOTA_LIMITS = {
        'read_requests': 25000,      # per day
        'write_requests': 500,        # per day (sensitive)
        'upload_requests': 1000000000, # bytes per day
    }
    
    def __init__(self):
        self.request_counts = {
            'read': 0,
            'write': 0,
            'upload': 0
        }
        self.last_reset = time.time()
    
    def check_rate_limit(self, operation_type: str):
        """Check if operation is within rate limits"""
        # Reset counters daily
        if time.time() - self.last_reset > 86400:
            self.request_counts = {'read': 0, 'write': 0, 'upload': 0}
            self.last_reset = time.time()
        
        # Check limits
        if operation_type == 'read' and self.request_counts['read'] >= self.QUOTA_LIMITS['read_requests']:
            raise GmailRateLimitError("Daily read quota exceeded")
        
        if operation_type == 'write' and self.request_counts['write'] >= self.QUOTA_LIMITS['write_requests']:
            raise GmailRateLimitError("Daily write quota exceeded")
    
    def increment(self, operation_type: str):
        """Increment operation counter"""
        self.request_counts[operation_type] += 1

def with_retry(max_retries=3, backoff_factor=2):
    """Decorator for retry with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        raise
                    
                    # Exponential backoff
                    wait_time = backoff_factor ** retries
                    time.sleep(wait_time)
            
            return None
        return wrapper
    return decorator
```

---

## 12. Security Considerations

### 12.1 Token Storage

```python
# Security best practices for token storage

import keyring
import json
from cryptography.fernet import Fernet

class SecureTokenStorage:
    """Secure token storage using Windows Credential Manager"""
    
    SERVICE_NAME = "OpenClawGmailAgent"
    
    def __init__(self):
        # Generate or retrieve encryption key
        self.key = self._get_or_create_key()
        self.cipher = Fernet(self.key)
    
    def _get_or_create_key(self) -> bytes:
        """Get or create encryption key"""
        key = keyring.get_password(self.SERVICE_NAME, "encryption_key")
        if not key:
            key = Fernet.generate_key().decode()
            keyring.set_password(self.SERVICE_NAME, "encryption_key", key)
        return key.encode()
    
    def store_token(self, token_data: dict):
        """Encrypt and store token"""
        encrypted = self.cipher.encrypt(
            json.dumps(token_data).encode()
        )
        keyring.set_password(
            self.SERVICE_NAME,
            "gmail_token",
            encrypted.decode()
        )
    
    def retrieve_token(self) -> dict:
        """Retrieve and decrypt token"""
        encrypted = keyring.get_password(self.SERVICE_NAME, "gmail_token")
        if encrypted:
            decrypted = self.cipher.decrypt(encrypted.encode())
            return json.loads(decrypted.decode())
        return None
```

### 12.2 Security Checklist

- [ ] Store credentials in Windows Credential Manager or encrypted file
- [ ] Use HTTPS for all webhook endpoints
- [ ] Implement request validation for webhooks
- [ ] Rotate OAuth tokens periodically
- [ ] Use minimal required scopes
- [ ] Implement proper logging without exposing sensitive data
- [ ] Handle token refresh automatically
- [ ] Revoke tokens on logout/uninstall

---

## Appendix A: Quick Reference

### A.1 Common Operations

```python
# Initialize client
from gmail_client import GmailClient

client = GmailClient(
    client_secrets_file='credentials/client_secret.json',
    token_file='credentials/token.json'
)

# List unread messages
messages = client.messages.list_messages(
    query='is:unread',
    max_results=10
)

# Send email
sent = client.messages.send_message(
    to='recipient@example.com',
    subject='Hello from AI Agent',
    body='This is a test email.',
    body_type='plain'
)

# Create label
label = client.labels.create_label(
    name='AI-Processed',
    background_color='#16a766',
    text_color='#ffffff'
)

# Get thread conversation
conversation = client.threads.get_conversation_view(thread_id='...')

# Download attachments
attachments = client.attachments.download_all_attachments(
    message_id='...',
    save_path='downloads/'
)
```

### A.2 Gmail Search Query Examples

```python
# Unread emails from specific sender
query = 'is:unread from:boss@company.com'

# Emails with attachments from last week
query = 'has:attachment newer_than:7d'

# Important emails not in inbox
query = 'is:important -in:inbox'

# Large emails
query = 'larger:10M'

# Emails in specific label
query = 'label:Projects'
```

---

**Document End**

*This specification provides comprehensive guidance for implementing Gmail API integration in a Windows 10 AI agent system.*
