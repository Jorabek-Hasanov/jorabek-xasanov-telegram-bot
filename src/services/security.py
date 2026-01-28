# Security Service
# ============================================================================
# Security utilities including encryption and prompt injection protection
# Implements Fernet symmetric encryption with dynamic salt for maximum security
# ============================================================================

import os
import re
import secrets
from dataclasses import dataclass
from typing import List, Optional, Tuple

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

import base64

from src.core.config import get_settings
from src.core.logging import logger


@dataclass
class SecurityCheckResult:
    """Result of security check."""
    safe: bool
    violations: List[str]
    sanitized_text: str


class EncryptionService:
    """
    Symmetric encryption service using Fernet with dynamic salt.
    
    Features:
    - Dynamic salt generation per encryption operation (prevents Mass Decryption)
    - Secure PBKDF2 key derivation
    - Salt stored with ciphertext for decryption
    - Base64 encoding for database storage
    
    Architecture:
    - Each encryption generates a unique 16-byte salt
    - Salt is prepended to the encrypted data
    - Format: base64(salt + encrypted_data)
    
    Example:
        ```python
        service = EncryptionService(master_password="secret")
        encrypted = service.encrypt("phone_number")
        decrypted = service.decrypt(encrypted)
        ```
    """
    
    # Salt length in bytes (16 bytes = 128 bits)
    SALT_LENGTH = 16
    
    # PBKDF2 parameters (OWASP recommended)
    PBKDF2_ITERATIONS = 480000
    
    def __init__(self, master_password: str = "") -> None:
        """
        Initialize encryption service.
        
        Args:
            master_password: Master password for key derivation
        """
        self.settings = get_settings().security
        self._master_password = master_password or self.settings.encryption_key
        
        if not self._master_password:
            self._fernet: Optional[Fernet] = None
            logger.warning("No encryption key provided, encryption disabled")
        else:
            self._fernet = None  # Will be created per-operation with dynamic salt
            logger.info("Encryption service initialized with dynamic salt support")
    
    def _derive_fernet_key(self, salt: bytes) -> Fernet:
        """
        Derive Fernet key from master password and salt.
        
        Args:
            salt: Dynamic salt bytes (16 bytes)
            
        Returns:
            Fernet instance for encryption/decryption
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.PBKDF2_ITERATIONS,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self._master_password.encode()))
        return Fernet(key)
    
    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt sensitive data with dynamic salt.
        
        Each call generates a unique salt, preventing Mass Decryption attacks.
        
        Args:
            plaintext: Text to encrypt
            
        Returns:
            Base64-encoded data containing: salt + encrypted_data
            
        Raises:
            RuntimeError: If encryption is not configured
        """
        if not self._master_password:
            raise RuntimeError("Encryption not configured")
        
        # Generate cryptographically secure random salt
        salt = secrets.token_bytes(self.SALT_LENGTH)
        
        # Derive key with dynamic salt
        fernet = self._derive_fernet_key(salt)
        
        # Encrypt data
        encrypted = fernet.encrypt(plaintext.encode())
        
        # Prepend salt to encrypted data
        salted_data = salt + encrypted
        
        # Base64 encode for storage
        return base64.urlsafe_b64encode(salted_data).decode()
    
    def decrypt(self, ciphertext: str) -> str:
        """
        Decrypt data using stored dynamic salt.
        
        Args:
            ciphertext: Base64-encoded data (salt + encrypted_data)
            
        Returns:
            Decrypted plaintext
            
        Raises:
            RuntimeError: If encryption is not configured
            ValueError: If decryption fails
        """
        if not self._master_password:
            raise RuntimeError("Encryption not configured")
        
        try:
            # Decode from base64
            salted_data = base64.urlsafe_b64decode(ciphertext.encode())
            
            # Extract salt (first 16 bytes) and encrypted data
            salt = salted_data[:self.SALT_LENGTH]
            encrypted = salted_data[self.SALT_LENGTH:]
            
            # Derive key with extracted salt and decrypt
            fernet = self._derive_fernet_key(salt)
            decrypted = fernet.decrypt(encrypted)
            
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError("Failed to decrypt data - invalid key or corrupted data")
    
    def encrypt_dict(self, data: dict) -> dict:
        """Encrypt all string values in a dictionary."""
        if not self._master_password:
            return data
        
        encrypted = {}
        for key, value in data.items():
            if isinstance(value, str):
                encrypted[key] = self.encrypt(value)
            else:
                encrypted[key] = value
        return encrypted
    
    def decrypt_dict(self, data: dict) -> dict:
        """Decrypt all values in a dictionary."""
        if not self._master_password:
            return data
        
        decrypted = {}
        for key, value in data.items():
            if isinstance(value, str):
                try:
                    decrypted[key] = self.decrypt(value)
                except (ValueError, Exception):
                    decrypted[key] = value  # Not encrypted, keep as-is
            else:
                decrypted[key] = value
        return decrypted


class SecurityService:
    """
    Security service for input validation and protection.
    
    Features:
    - Prompt injection detection
    - Input sanitization
    - XSS prevention
    - Special character filtering
    
    Example:
        ```python
        service = SecurityService()
        result = service.check_prompt_injection("ignore previous instructions")
        print(result.safe)  # False
        ```
    """
    
    def __init__(self) -> None:
        """Initialize security service."""
        self.settings = get_settings().security
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for performance."""
        self.injection_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.settings.prompt_injection_patterns
        ]
        
        # XSS patterns
        self.xss_patterns = [
            re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
            re.compile(r'javascript:', re.IGNORECASE),
            re.compile(r'on\w+\s*=', re.IGNORECASE),
            re.compile(r'<iframe[^>]*>.*?</iframe>', re.IGNORECASE | re.DOTALL),
        ]
        
        # Control characters
        self.control_char_pattern = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
    
    def check_prompt_injection(self, text: str) -> SecurityCheckResult:
        """
        Check text for prompt injection attempts.
        
        Args:
            text: User input to check
            
        Returns:
            SecurityCheckResult with violations and sanitized text
        """
        violations = []
        sanitized = text
        
        # Check injection patterns
        for i, pattern in enumerate(self.injection_patterns):
            if pattern.search(text):
                violations.append(f"Prompt injection pattern detected: {pattern.pattern}")
                # Remove the matched text
                sanitized = pattern.sub('', sanitized)
        
        # Log security event
        if violations:
            logger.log_security_event(
                event_type="prompt_injection_attempt",
                user_id=None,
                details={"violations": violations, "original_length": len(text)}
            )
        
        return SecurityCheckResult(
            safe=len(violations) == 0,
            violations=violations,
            sanitized_text=sanitized.strip(),
        )
    
    def sanitize_input(self, text: str) -> str:
        """
        Sanitize user input for safety.
        
        Removes:
        - HTML/script tags (XSS prevention)
        - Control characters
        - Excessive whitespace
        
        Args:
            text: Raw user input
            
        Returns:
            Sanitized text
        """
        # Remove HTML tags
        sanitized = re.sub(r'<[^>]+>', '', text)
        
        # Remove control characters
        sanitized = self.control_char_pattern.sub('', sanitized)
        
        # Remove excessive newlines
        sanitized = re.sub(r'\n{3,}', '\n\n', sanitized)
        
        # Limit length
        max_length = 10000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()
    
    def check_xss(self, text: str) -> Tuple[bool, str]:
        """
        Check for XSS attempts.
        
        Args:
            text: Text to check
            
        Returns:
            Tuple of (is_safe, sanitized_text)
        """
        violations = []
        sanitized = text
        
        for pattern in self.xss_patterns:
            if pattern.search(text):
                violations.append(f"XSS pattern detected: {pattern.pattern}")
                sanitized = pattern.sub('', sanitized)
        
        if violations:
            logger.log_security_event(
                event_type="xss_attempt",
                user_id=None,
                details={"violations": violations}
            )
        
        return len(violations) == 0, sanitized
    
    def detect_language(self, text: str) -> str:
        """
        Detect language from text.
        
        Supports: Uzbek, Russian, English, German, French.
        
        Args:
            text: Text to analyze
            
        Returns:
            Detected language name
        """
        if not text:
            return "English"
        
        text_lower = text.lower().strip()
        
        # Uzbek indicators
        uzbek_markers = [
            "ассалому алайкум", "алайкум ассалом", "раҳмат", "яхши",
            "узбек", "узбекча", "ҳаммаси", "биласанми", "менинг",
            "бу", "шу", "у", "олдин", "кейин", "кандай", "нима"
        ]
        
        # Russian indicators
        russian_markers = [
            "привет", "здравствуй", "как дела", "спасибо", "пожалуйста",
            "россия", "русский", "можно", "подскажи", "помоги"
        ]
        
        # German indicators
        german_markers = [
            "hallo", "gut", "danke", "bitte", "wie", "deutschland"
        ]
        
        # French indicators
        french_markers = [
            "bonjour", "merci", "svp", "comment", "français", "ça va"
        ]
        
        for marker in uzbek_markers:
            if marker in text_lower:
                return "Uzbek"
        
        for marker in russian_markers:
            if marker in text_lower:
                return "Russian"
        
        for marker in german_markers:
            if marker in text_lower:
                return "German"
        
        for marker in french_markers:
            if marker in text_lower:
                return "French"
        
        return "English"
