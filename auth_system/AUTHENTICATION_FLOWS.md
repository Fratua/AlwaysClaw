# Authentication Flows Documentation

## For Windows 10 OpenClaw AI Agent Framework

This document provides detailed visual representations of all authentication flows supported by the authentication system.

---

## Table of Contents

1. [Form-Based Authentication Flow](#1-form-based-authentication-flow)
2. [OAuth 2.0 Authorization Code Flow](#2-oauth-20-authorization-code-flow)
3. [OAuth 2.0 Client Credentials Flow](#3-oauth-20-client-credentials-flow)
4. [OAuth 2.0 Device Code Flow](#4-oauth-20-device-code-flow)
5. [JWT Token Lifecycle](#5-jwt-token-lifecycle)
6. [MFA (TOTP) Flow](#6-mfa-totp-flow)
7. [SAML 2.0 SSO Flow](#7-saml-20-sso-flow)
8. [OpenID Connect Flow](#8-openid-connect-flow)
9. [Session Persistence Flow](#9-session-persistence-flow)

---

## 1. Form-Based Authentication Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Form-Based Authentication                            │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐                    ┌──────────────┐
    │   AI Agent   │                    │    Target    │
    │              │                    │   Website    │
    └──────┬───────┘                    └──────┬───────┘
           │                                    │
           │ 1. Navigate to login page          │
           │───────────────────────────────────>│
           │                                    │
           │ 2. Return login form HTML          │
           │<───────────────────────────────────│
           │                                    │
           │ 3. Detect login form               │
           │    (AI Form Detection)             │
           │────────┐                         │
           │        │                         │
           │<───────┘                         │
           │                                    │
           │ 4. Retrieve credentials            │
           │    from vault                      │
           │────────┐                         │
           │        │                         │
           │<───────┘                         │
           │                                    │
           │ 5. Fill username/password          │
           │    (with human-like delays)        │
           │───────────────────────────────────>│
           │                                    │
           │ 6. Submit form                     │
           │───────────────────────────────────>│
           │                                    │
           │ 7. Check for MFA/CAPTCHA           │
           │<───────────────────────────────────│
           │                                    │
      ┌────┴────┐                               │
      │  MFA?   │                               │
      └────┬────┘                               │
    Yes    │    No                              │
         ┌─┴─┐                                  │
         │   │                                  │
         ▼   ▼                                  │
    ┌────────┐  ┌────────┐                     │
    │ Handle │  │ Check  │                     │
    │  MFA   │  │ Success│                     │
    └────┬───┘  └───┬────┘                     │
         │          │                          │
         └────┬─────┘                          │
              │                                 │
              ▼                                 │
    ┌──────────────────┐                       │
    │  8. Sync cookies │                       │
    │     to cookie jar│                       │
    └────────┬─────────┘                       │
             │                                 │
             ▼                                 │
    ┌──────────────────┐                       │
    │  9. Update auth  │                       │
    │     context      │                       │
    └────────┬─────────┘                       │
             │                                 │
             ▼                                 │
    ┌──────────────────┐                       │
    │  10. Return      │                       │
    │      AuthResult  │                       │
    └──────────────────┘                       │
```

### Flow Steps:

1. **Navigate** - Browser navigates to login page
2. **Load Form** - Page returns HTML with login form
3. **Detect Form** - AI analyzes page to identify login form fields
4. **Get Credentials** - Retrieve encrypted credentials from vault
5. **Fill Form** - Enter username/password with realistic timing
6. **Submit** - Click submit or press Enter
7. **Check Response** - Analyze response for MFA/CAPTCHA requirements
8. **Handle MFA** - If required, process TOTP/SMS/Email MFA
9. **Sync Cookies** - Save session cookies to encrypted jar
10. **Update Context** - Mark authentication as successful

---

## 2. OAuth 2.0 Authorization Code Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OAuth 2.0 Authorization Code Flow                        │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐        ┌──────────────┐        ┌──────────────────┐
    │   AI Agent   │        │     User     │        │  OAuth Provider  │
    │   (Client)   │        │   (Browser)  │        │  (Google, etc.)  │
    └──────┬───────┘        └──────┬───────┘        └────────┬─────────┘
           │                       │                         │
           │ 1. Generate state     │                         │
           │    & PKCE params      │                         │
           │──────┐                │                         │
           │      │                │                         │
           │<─────┘                │                         │
           │                       │                         │
           │ 2. Build auth URL     │                         │
           │    with params        │                         │
           │──────┐                │                         │
           │      │                │                         │
           │<─────┘                │                         │
           │                       │                         │
           │ 3. Redirect user      │                         │
           │──────────────────────>│                         │
           │                       │                         │
           │                       │ 4. Navigate to IdP      │
           │                       │────────────────────────>│
           │                       │                         │
           │                       │ 5. User authenticates   │
           │                       │    (if not logged in)   │
           │                       │<───────────────────────>│
           │                       │                         │
           │                       │ 6. User consents to     │
           │                       │    requested scopes     │
           │                       │<───────────────────────>│
           │                       │                         │
           │                       │ 7. Redirect with        │
           │                       │    authorization code   │
           │                       │<────────────────────────│
           │                       │                         │
           │ 8. Code received      │                         │
           │<──────────────────────│                         │
           │                       │                         │
           │ 9. Exchange code      │                         │
           │    for tokens         │                         │
           │─────────────────────────────────────────────────>│
           │                       │                         │
           │ 10. Tokens returned   │                         │
           │<────────────────────────────────────────────────│
           │    (access_token,     │                         │
           │     refresh_token,    │                         │
           │     id_token)         │                         │
           │                       │                         │
           │ 11. Validate tokens   │                         │
           │    & store securely   │                         │
           │──────┐                │                         │
           │      │                │                         │
           │<─────┘                │                         │
           │                       │                         │
```

### Flow Steps:

1. **Generate State** - Create random state for CSRF protection
2. **Generate PKCE** - Create code_verifier and code_challenge
3. **Build Auth URL** - Construct authorization endpoint URL
4. **Redirect** - Send user to OAuth provider
5. **User Login** - User authenticates with provider
6. **Consent** - User approves requested permissions
7. **Authorization Code** - Provider redirects with code
8. **Receive Code** - Extract code from callback
9. **Token Exchange** - POST code to token endpoint
10. **Receive Tokens** - Get access_token, refresh_token, id_token
11. **Store Securely** - Encrypt and store tokens

---

## 3. OAuth 2.0 Client Credentials Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OAuth 2.0 Client Credentials Flow                        │
│                          (Machine-to-Machine)                               │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐                              ┌──────────────────┐
    │   AI Agent   │                              │  OAuth Provider  │
    │   (Client)   │                              │  (API Service)   │
    └──────┬───────┘                              └────────┬─────────┘
           │                                              │
           │ 1. Prepare token request                     │
           │    grant_type=client_credentials             │
           │    client_id=xxx                             │
           │    client_secret=xxx                         │
           │    scope=requested_scope                     │
           │──────┐                                       │
           │      │                                       │
           │<─────┘                                       │
           │                                              │
           │ 2. POST to token endpoint                    │
           │─────────────────────────────────────────────>│
           │                                              │
           │ 3. Provider validates client credentials     │
           │    and scope permissions                     │
           │    ┌─────────────────┐                       │
           │    │ Validate client │                       │
           │    │ Check scopes    │                       │
           │    └─────────────────┘                       │
           │                                              │
           │ 4. Return access token                       │
           │<─────────────────────────────────────────────│
           │    (no refresh token in this flow)           │
           │                                              │
           │ 5. Use access token for API calls            │
           │──────┐                                       │
           │      │                                       │
           │<─────┘                                       │
           │                                              │
           │ 6. Token expires                             │
           │    (typically 1 hour)                        │
           │                                              │
           │ 7. Request new token                         │
           │    (repeat from step 1)                      │
           │─────────────────────────────────────────────>│
           │                                              │
```

### Flow Steps:

1. **Prepare Request** - Build token request with client credentials
2. **POST to Token Endpoint** - Send client_id and client_secret
3. **Validate** - Provider validates credentials and scope
4. **Return Token** - Access token returned (no refresh token)
5. **Use Token** - Include in API requests
6. **Token Expires** - Access token has limited lifetime
7. **Request New** - Repeat flow for new token

---

## 4. OAuth 2.0 Device Code Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      OAuth 2.0 Device Code Flow                             │
│                     (For Input-Constrained Devices)                         │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐         ┌──────────────┐         ┌──────────────────┐
    │   AI Agent   │         │     User     │         │  OAuth Provider  │
    │  (Headless)  │         │  (Secondary  │         │  (Google, etc.)  │
    │              │         │   Device)    │         │                  │
    └──────┬───────┘         └──────┬───────┘         └────────┬─────────┘
           │                        │                          │
           │ 1. Request device code │                          │
           │───────────────────────────────────────────────────>│
           │                        │                          │
           │ 2. Return device code  │                          │
           │    response:           │                          │
           │    - device_code       │                          │
           │    - user_code         │                          │
           │    - verification_uri  │                          │
           │    - interval          │                          │
           │    - expires_in        │                          │
           │<───────────────────────────────────────────────────│
           │                        │                          │
           │ 3. Display user_code   │                          │
           │    & verification_uri  │                          │
           │───────────────────────>│                          │
           │                        │                          │
           │                        │ 4. User visits           │
           │                        │    verification_uri      │
           │                        │─────────────────────────>│
           │                        │                          │
           │                        │ 5. User enters user_code │
           │                        │<────────────────────────>│
           │                        │                          │
           │                        │ 6. User authenticates    │
           │                        │    and consents          │
           │                        │<────────────────────────>│
           │                        │                          │
           │ 7. Poll token endpoint │                          │
           │    (every N seconds)   │                          │
           │───────────────────────────────────────────────────>│
           │                        │                          │
           │ 8. authorization_pending│                         │
           │<───────────────────────────────────────────────────│
           │                        │                          │
           │ 9. Continue polling... │                          │
           │    [Repeat until user  │                          │
           │     completes auth]    │                          │
           │                        │                          │
           │ 10. Tokens returned    │                          │
           │<───────────────────────────────────────────────────│
           │                        │                          │
```

### Flow Steps:

1. **Request Device Code** - POST to device authorization endpoint
2. **Receive Codes** - Get device_code, user_code, verification_uri
3. **Display to User** - Show user_code and URL
4. **User Visits URL** - User opens verification page on another device
5. **Enter Code** - User inputs the user_code
6. **Authenticate** - User logs in and approves
7. **Poll for Token** - Device polls token endpoint
8. **Pending Response** - "authorization_pending" until user completes
9. **Continue Polling** - Poll at specified interval
10. **Receive Tokens** - Access and refresh tokens returned

---

## 5. JWT Token Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          JWT Token Lifecycle                                │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │ Token Issued │
                              │   (Auth)     │
                              └──────┬───────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │        Access Token            │
                    │  ┌────────────────────────┐    │
                    │  │ Header: alg=RS256      │    │
                    │  │ Payload: sub, exp, etc │    │
                    │  │ Signature: RSA-SHA256  │    │
                    │  └────────────────────────┘    │
                    │         15 min TTL             │
                    └─────────────┬──────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │   API Request   │ │   API Request   │ │   API Request   │
    │   (Bearer)      │ │   (Bearer)      │ │   (Bearer)      │
    └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
             │                   │                   │
             │                   │                   │
             ▼                   ▼                   ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │    Success      │ │    Success      │ │  Token Expired  │
    │                 │ │                 │ │  (15 min passed)│
    └─────────────────┘ └─────────────────┘ └────────┬────────┘
                                                       │
                                                       ▼
                                          ┌─────────────────────┐
                                          │  Refresh Required   │
                                          └──────────┬──────────┘
                                                     │
                                                     ▼
                                          ┌─────────────────────┐
                                          │ POST /token         │
                                          │ grant_type=refresh  │
                                          │ refresh_token=xxx   │
                                          └──────────┬──────────┘
                                                     │
                                                     ▼
                                          ┌─────────────────────┐
                                          │  New Token Pair     │
                                          │  (Rotation)         │
                                          │  - New access_token │
                                          │  - New refresh_token│
                                          │  - Old refresh      │
                                          │    marked used      │
                                          └──────────┬──────────┘
                                                     │
                                                     ▼
                                          ┌─────────────────────┐
                                          │  Continue with      │
                                          │  new access token   │
                                          └─────────────────────┘
```

### Token Lifecycle:

1. **Token Issued** - After authentication, tokens are generated
2. **Access Token Usage** - Used for API calls (15 min TTL)
3. **Multiple Requests** - Same token used for multiple API calls
4. **Token Expires** - After 15 minutes, token is invalid
5. **Refresh Required** - Use refresh token to get new access token
6. **Token Rotation** - New refresh token issued, old one invalidated
7. **Continue** - Use new access token for subsequent requests

---

## 6. MFA (TOTP) Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Multi-Factor Authentication (TOTP)                       │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐                    ┌──────────────┐
    │   AI Agent   │                    │    Target    │
    │              │                    │   Service    │
    └──────┬───────┘                    └──────┬───────┘
           │                                    │
           │ 1. Submit username/password        │
           │───────────────────────────────────>│
           │                                    │
           │ 2. Return MFA challenge page       │
           │<───────────────────────────────────│
           │    "Enter 6-digit code"            │
           │                                    │
           │ 3. Detect MFA type                 │
           │    (TOTP/SMS/Email)                │
           │──────┐                            │
           │      │                            │
           │<─────┘                            │
           │                                    │
           │ 4. Retrieve TOTP secret            │
           │    from vault                      │
           │──────┐                            │
           │      │                            │
           │<─────┘                            │
           │                                    │
           │ 5. Generate TOTP code              │
           │    (time-based algorithm)          │
           │    TOTP = HOTP(K, T)               │
           │    where T = (current_time - T0)/X │
           │──────┐                            │
           │      │                            │
           │<─────┘                            │
           │                                    │
           │ 6. Submit TOTP code                │
           │───────────────────────────────────>│
           │                                    │
           │ 7. Validate code                   │
           │    (server-side verification)      │
           │    ┌─────────────────┐             │
           │    │ Verify TOTP     │             │
           │    │ Check time drift│             │
           │    │ Allow ±1 window │             │
           │    └─────────────────┘             │
           │                                    │
           │ 8. Authentication successful       │
           │<───────────────────────────────────│
           │    Session cookie returned         │
           │                                    │
```

### TOTP Algorithm:

```
TOTP = HOTP(K, T)

Where:
- K = Shared secret key (Base32 encoded)
- T = (Current_Unix_Time - T0) / X
- T0 = Unix time to start counting time steps (default 0)
- X = Time step in seconds (default 30)

Example:
- Secret: JBSWY3DPEHPK3PXP
- Time: 1640995200 (2022-01-01 00:00:00 UTC)
- T = 1640995200 / 30 = 54699840
- TOTP = HOTP(JBSWY3DPEHPK3PXP, 54699840) = 123456
```

---

## 7. SAML 2.0 SSO Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SAML 2.0 SSO Flow                                  │
│                    (Service Provider Initiated)                             │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐        ┌──────────────┐        ┌──────────────────┐
    │   AI Agent   │        │     User     │        │  Identity        │
    │  (Service    │        │   (Browser)  │        │  Provider (IdP)  │
    │  Provider)   │        │              │        │  (Azure AD, etc) │
    └──────┬───────┘        └──────┬───────┘        └────────┬─────────┘
           │                       │                         │
           │ 1. Access protected   │                         │
           │    resource           │                         │
           │<──────────────────────│                         │
           │                       │                         │
           │ 2. No valid session   │                         │
           │    Generate SAML      │                         │
           │    AuthnRequest       │                         │
           │──────┐                │                         │
           │      │                │                         │
           │      │                │                         │
           │      ▼                │                         │
           │  ┌─────────────────┐  │                         │
           │  │ Build SAML XML: │  │                         │
           │  │ <AuthnRequest>  │  │                         │
           │  │   ID="_xxx"     │  │                         │
           │  │   IssueInstant  │  │                         │
           │  │   Destination   │  │                         │
           │  │   <Issuer>SP</> │  │                         │
           │  │ </AuthnRequest>  │  │                         │
           │  └─────────────────┘  │                         │
           │      │                │                         │
           │      │                │                         │
           │<─────┘                │                         │
           │                       │                         │
           │ 3. Sign & encode      │                         │
           │    (deflate + base64) │                         │
           │──────┐                │                         │
           │      │                │                         │
           │<─────┘                │                         │
           │                       │                         │
           │ 4. Redirect to IdP    │                         │
           │    with SAMLRequest   │                         │
           │──────────────────────>│                         │
           │                       │                         │
           │                       │ 5. POST to IdP SSO URL  │
           │                       │────────────────────────>│
           │                       │    SAMLRequest=xxx      │
           │                       │    RelayState=xxx       │
           │                       │                         │
           │                       │ 6. Validate AuthnRequest│
           │                       │    ┌─────────────────┐  │
           │                       │    │ Verify signature│  │
           │                       │    │ Check recipient │  │
           │                       │    │ Validate times  │  │
           │                       │    └─────────────────┘  │
           │                       │                         │
           │                       │ 7. User authenticates   │
           │                       │    (if needed)          │
           │                       │<───────────────────────>│
           │                       │                         │
           │                       │ 8. Generate SAML        │
           │                       │    Response (Assertion) │
           │                       │──────┐                  │
           │                       │      │                  │
           │                       │      ▼                  │
           │                       │  ┌─────────────────┐    │
           │                       │  │ <Response>      │    │
           │                       │  │   <Assertion>   │    │
           │                       │  │     <Subject>   │    │
           │                       │  │       <NameID>  │    │
           │                       │  │     </Subject>  │    │
           │                       │  │     <Conditions>│    │
           │                       │  │     <AuthnStmt> │    │
           │                       │  │     <AttrStmt>  │    │
           │                       │  │   </Assertion>  │    │
           │                       │  │ </Response>     │    │
           │                       │  └─────────────────┘    │
           │                       │      │                  │
           │                       │<─────┘                  │
           │                       │                         │
           │                       │ 9. POST to SP ACS       │
           │                       │<────────────────────────│
           │                       │    SAMLResponse=xxx     │
           │                       │    RelayState=xxx       │
           │                       │                         │
           │ 10. Validate Response │                         │
           │     ┌─────────────────┐                        │
           │     │ Verify signature│                        │
           │     │ Check conditions│                        │
           │     │ Validate issuer │                        │
           │     │ Extract attrs   │                        │
           │     └─────────────────┘                        │
           │──────┐                                       │
           │      │                                       │
           │<─────┘                                       │
           │                                              │
           │ 11. Create local session                     │
           │     Set session cookie                       │
           │──────┐                                       │
           │      │                                       │
           │<─────┘                                       │
           │                                              │
           │ 12. Redirect to original resource            │
           │──────────────────────>│                       │
           │                                              │
```

### SAML Assertion Structure:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<saml2:Assertion 
    xmlns:saml2="urn:oasis:names:tc:SAML:2.0:assertion"
    ID="_a75adf55-01d7-40cc-929f-dbd8372ebdfc"
    IssueInstant="2025-01-15T10:30:00Z"
    Version="2.0">
    
    <saml2:Issuer>https://idp.example.com</saml2:Issuer>
    
    <saml2:Subject>
        <saml2:NameID Format="urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress">
            user@example.com
        </saml2:NameID>
        <saml2:SubjectConfirmation Method="urn:oasis:names:tc:SAML:2.0:cm:bearer">
            <saml2:SubjectConfirmationData 
                NotOnOrAfter="2025-01-15T10:35:00Z"
                Recipient="https://sp.example.com/saml/acs"/>
        </saml2:SubjectConfirmation>
    </saml2:Subject>
    
    <saml2:Conditions NotBefore="2025-01-15T10:30:00Z" NotOnOrAfter="2025-01-15T11:30:00Z">
        <saml2:AudienceRestriction>
            <saml2:Audience>https://sp.example.com</saml2:Audience>
        </saml2:AudienceRestriction>
    </saml2:Conditions>
    
    <saml2:AuthnStatement 
        AuthnInstant="2025-01-15T10:30:00Z"
        SessionIndex="_session_index_value">
        <saml2:AuthnContext>
            <saml2:AuthnContextClassRef>
                urn:oasis:names:tc:SAML:2.0:ac:classes:PasswordProtectedTransport
            </saml2:AuthnContextClassRef>
        </saml2:AuthnContext>
    </saml2:AuthnStatement>
    
    <saml2:AttributeStatement>
        <saml2:Attribute Name="email">
            <saml2:AttributeValue>user@example.com</saml2:AttributeValue>
        </saml2:Attribute>
        <saml2:Attribute Name="firstName">
            <saml2:AttributeValue>John</saml2:AttributeValue>
        </saml2:Attribute>
        <saml2:Attribute Name="lastName">
            <saml2:AttributeValue>Doe</saml2:AttributeValue>
        </saml2:Attribute>
        <saml2:Attribute Name="groups">
            <saml2:AttributeValue>Admins</saml2:AttributeValue>
            <saml2:AttributeValue>Users</saml2:AttributeValue>
        </saml2:Attribute>
    </saml2:AttributeStatement>
    
</saml2:Assertion>
```

---

## 8. OpenID Connect Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OpenID Connect Flow                                 │
│                         (Authorization Code)                                │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐        ┌──────────────┐        ┌──────────────────┐
    │   AI Agent   │        │     User     │        │  OpenID Provider │
    │   (Client)   │        │   (Browser)  │        │  (Google, etc.)  │
    └──────┬───────┘        └──────┬───────┘        └────────┬─────────┘
           │                       │                         │
           │ 1. Generate state     │                         │
           │    & nonce            │                         │
           │    Generate PKCE      │                         │
           │    (code_verifier)    │                         │
           │──────┐                │                         │
           │      │                │                         │
           │      ▼                │                         │
           │  ┌─────────────────┐  │                         │
           │  │ State: random   │  │                         │
           │  │ Nonce: random   │  │                         │
           │  │ Code Challenge: │  │                         │
           │  │   SHA256(verif) │  │                         │
           │  └─────────────────┘  │                         │
           │      │                │                         │
           │<─────┘                │                         │
           │                       │                         │
           │ 2. Build auth URL     │                         │
           │    /authorize?        │                         │
           │    response_type=code │                         │
           │    client_id=xxx      │                         │
           │    redirect_uri=xxx   │                         │
           │    scope=openid email │                         │
           │    state=xxx          │                         │
           │    nonce=xxx          │                         │
           │    code_challenge=xxx │                         │
           │    code_challenge_meth│                         │
           │      =S256            │                         │
           │──────┐                │                         │
           │      │                │                         │
           │<─────┘                │                         │
           │                       │                         │
           │ 3. Redirect user      │                         │
           │──────────────────────>│                         │
           │                       │                         │
           │                       │ 4. Navigate to IdP      │
           │                       │────────────────────────>│
           │                       │                         │
           │                       │ 5. User authenticates   │
           │                       │<───────────────────────>│
           │                       │                         │
           │                       │ 6. User consents to     │
           │                       │    requested scopes     │
           │                       │<───────────────────────>│
           │                       │                         │
           │                       │ 7. Redirect with        │
           │                       │    authorization code   │
           │                       │<────────────────────────│
           │                       │                         │
           │ 8. Code received      │                         │
           │    (verify state)     │                         │
           │<──────────────────────│                         │
           │                       │                         │
           │ 9. Exchange code      │                         │
           │    POST /token        │                         │
           │    grant_type=auth_cod│                         │
           │    code=xxx           │                         │
           │    redirect_uri=xxx   │                         │
           │    code_verifier=xxx  │                         │
           │─────────────────────────────────────────────────>│
           │                       │                         │
           │ 10. Return tokens     │                         │
           │<────────────────────────────────────────────────│
           │    {                  │                         │
           │      access_token,    │                         │
           │      refresh_token,   │                         │
           │      id_token (JWT),  │                         │
           │      token_type,      │                         │
           │      expires_in       │                         │
           │    }                  │                         │
           │                       │                         │
           │ 11. Validate ID Token│                         │
           │     ┌─────────────────┐                        │
           │     │ Verify signature│                        │
           │     │ Check issuer    │                        │
           │     │ Verify audience │                        │
           │     │ Check nonce     │                        │
           │     │ Validate exp    │                        │
           │     └─────────────────┘                        │
           │──────┐                                       │
           │      │                                       │
           │      ▼                                       │
           │  ┌─────────────────┐                         │
           │  │ ID Token Claims:│                         │
           │  │   sub (user ID) │                         │
           │  │   iss (issuer)  │                         │
           │  │   aud (audience)│                         │
           │  │   email         │                         │
           │  │   name          │                         │
           │  │   picture       │                         │
           │  └─────────────────┘                         │
           │      │                                       │
           │<─────┘                                       │
           │                                              │
           │ 12. Optional: Get UserInfo                  │
           │     GET /userinfo                           │
           │     Authorization: Bearer {access_token}    │
           │─────────────────────────────────────────────────>│
           │                                              │
           │ 13. Return user attributes                  │
           │<────────────────────────────────────────────────│
           │                                              │
```

### ID Token Structure:

```json
{
  "header": {
    "alg": "RS256",
    "kid": "key-id-123",
    "typ": "JWT"
  },
  "payload": {
    "iss": "https://accounts.google.com",
    "sub": "1234567890",
    "aud": "your-client-id.apps.googleusercontent.com",
    "exp": 1640998800,
    "iat": 1640995200,
    "auth_time": 1640995100,
    "nonce": "random-nonce-value",
    "email": "user@example.com",
    "email_verified": true,
    "name": "John Doe",
    "picture": "https://lh3.googleusercontent.com/...",
    "given_name": "John",
    "family_name": "Doe",
    "locale": "en"
  },
  "signature": "RSASHA256(base64url(header) + "." + base64url(payload), private_key)"
}
```

---

## 9. Session Persistence Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Session Persistence Flow                             │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
    │   AI Agent   │         │   Browser    │         │   Storage    │
    │              │         │   Context    │         │   (Disk)     │
    └──────┬───────┘         └──────┬───────┘         └──────┬───────┘
           │                        │                        │
           │                        │                        │
           │  INITIAL AUTHENTICATION                        │
           │                        │                        │
           │ 1. Navigate & Login   │                        │
           │──────────────────────>│                        │
           │                        │                        │
           │ 2. Auth successful    │                        │
           │<──────────────────────│                        │
           │                        │                        │
           │ 3. Capture session    │                        │
           │    - Cookies          │                        │
           │    - localStorage     │                        │
           │    - sessionStorage   │                        │
           │    - Tokens           │                        │
           │<──────────────────────│                        │
           │                        │                        │
           │ 4. Encrypt & save     │                        │
           │    session data       │                        │
           │───────────────────────────────────────────────>│
           │                        │                        │
           │                        │                        │
           │  SUBSEQUENT ACCESS (Later/Different Browser)   │
           │                        │                        │
           │                        │                        │
           │ 5. New browser        │                        │
           │    context created    │                        │
           │──────┐                │                        │
           │      │                │                        │
           │      │                │                        │
           │      ▼                │                        │
           │  ┌─────────────────┐  │                        │
           │  │ Load session    │  │                        │
           │  │ from storage    │  │                        │
           │  └─────────────────┘  │                        │
           │      │                │                        │
           │      │                │                        │
           │<─────┘                │                        │
           │                        │                        │
           │ 6. Decrypt session    │                        │
           │    data               │                        │
           │──────┐                │                        │
           │      │                │                        │
           │<─────┘                │                        │
           │                        │                        │
           │ 7. Restore to browser │                        │
           │    - Set cookies      │                        │
           │    - Set storage      │                        │
           │    - Inject tokens    │                        │
           │──────────────────────>│                        │
           │                        │                        │
           │ 8. Navigate to        │                        │
           │    protected page     │                        │
           │──────────────────────>│                        │
           │                        │                        │
           │ 9. Already            │                        │
           │    authenticated!     │                        │
           │<──────────────────────│                        │
           │                        │                        │
```

### Session Data Structure:

```json
{
  "version": "2.0",
  "session_id": "sess_a1b2c3d4e5f6",
  "service": "gmail",
  "created_at": "2025-01-15T10:30:00Z",
  "expires_at": "2025-01-15T18:30:00Z",
  "encryption": {
    "algorithm": "AES-256-GCM",
    "key_derivation": "Argon2id"
  },
  "cookies": [
    {
      "name": "SID",
      "value": "<encrypted>",
      "domain": ".google.com",
      "path": "/",
      "secure": true,
      "httpOnly": true,
      "sameSite": "Lax",
      "expires": "2025-07-15T10:30:00Z"
    }
  ],
  "local_storage": {
    "accounts.google.com": {
      "oauth_state": "<encrypted>",
      "user_prefs": "<encrypted>"
    }
  },
  "session_storage": {
    "mail.google.com": {
      "session_data": "<encrypted>"
    }
  },
  "tokens": {
    "access_token": "<encrypted>",
    "refresh_token": "<encrypted>",
    "id_token": "<encrypted>",
    "expires_at": "2025-01-15T11:30:00Z"
  },
  "metadata": {
    "user_agent": "Mozilla/5.0...",
    "ip_address": "192.168.1.1",
    "auth_method": "oauth"
  }
}
```

---

## Security Considerations

### Token Storage Security

| Component | Encryption | Key Management |
|-----------|------------|----------------|
| Cookies | AES-256-GCM | Master key from DPAPI |
| Credentials | AES-256-GCM + DPAPI | Windows Credential Manager |
| Sessions | AES-256-GCM | Master key from system entropy |
| Tokens | AES-256-GCM | Same as sessions |

### Network Security

- All OAuth/OIDC flows use HTTPS only
- PKCE prevents authorization code interception
- State parameter prevents CSRF attacks
- Nonce prevents replay attacks

### Session Security

- Short-lived access tokens (15 min)
- Refresh token rotation on each use
- Token reuse detection and revocation
- Secure, HttpOnly, SameSite cookies

---

## Error Handling

### Common Error Scenarios

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Error Handling Flow                                 │
└─────────────────────────────────────────────────────────────────────────────┘

Authentication Attempt
        │
        ▼
┌───────────────┐
│  Error Type   │
└───────┬───────┘
        │
   ┌────┴────┬────────────┬────────────┬────────────┐
   │         │            │            │            │
   ▼         ▼            ▼            ▼            ▼
Invalid   Network      MFA         Token       Session
Creds     Error       Required     Expired     Expired
   │         │            │            │            │
   ▼         ▼            ▼            ▼            ▼
Retry    Retry with   Prompt for   Refresh      Re-
w/ diff  backoff      MFA code     token        authenticate
 creds    or queue                              if refresh
                                                  fails
```

---

*Document Version: 1.0*
*Last Updated: January 2025*
