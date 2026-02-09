"""
OpenClaw User Modeling System

Unified API for user profile management, preference learning,
pattern recognition, personalization, adaptation, privacy, and
multi-user support.

Module layout:
  Part 1 (user_modeling_code_part1): Profile management & preference learning
    - ProfileVersion, UserProfile, UserProfileManager
    - PreferenceSignal, PreferenceLearningEngine
    - ContextualPreferenceLearner

  Part 2 (user_modeling_code_part2): Pattern recognition & personalization
    - BehaviorPattern, TemporalPatternDetector, SequentialPatternDetector
    - SemanticPatternDetector, PatternRecognitionOrchestrator
    - PersonalizationLevel, PersonalizationContext, PersonalizationResult
    - PersonalizationEngine, ResponsePersonalizer, UIPersonalizer

  Part 3 (user_modeling_code_part3): Adaptation, privacy & consent
    - AdaptationTrigger, AdaptationPriority, AdaptationAction
    - UserAdaptationEngine
    - DataSensitivity, DataCategory, DataField, PrivacySchema
    - PrivacyManager, ConsentManager

  Part 4 (user_modeling_code_part4): Feedback & multi-user support
    - FeedbackType, FeedbackContext, FeedbackItem
    - FeedbackCollectionSystem, FeedbackPromptingSystem
    - UserRole, Permission, ROLE_PERMISSIONS, UserSession
    - MultiUserManager, UserContextIsolation, SharedContextManager
"""

import sys
import os

# Ensure project root is on the path so sibling modules resolve
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# -- Part 1: Profile Management & Preference Learning -----------------------
from user_modeling_code_part1 import (
    ProfileVersion,
    UserProfile,
    UserProfileManager,
    PreferenceSignal,
    PreferenceLearningEngine,
    ContextualPreferenceLearner,
)

# -- Part 2: Pattern Recognition & Personalization --------------------------
from user_modeling_code_part2 import (
    BehaviorPattern,
    TemporalPatternDetector,
    SequentialPatternDetector,
    SemanticPatternDetector,
    PatternRecognitionOrchestrator,
    PersonalizationLevel,
    PersonalizationContext,
    PersonalizationResult,
    PersonalizationEngine,
    ResponsePersonalizer,
    UIPersonalizer,
)

# -- Part 3: Adaptation, Privacy & Consent -----------------------------------
from user_modeling_code_part3 import (
    AdaptationTrigger,
    AdaptationPriority,
    AdaptationAction,
    UserAdaptationEngine,
    DataSensitivity,
    DataCategory,
    DataField,
    PrivacySchema,
    PrivacyManager,
    ConsentManager,
)

# -- Part 4: Feedback & Multi-User Support -----------------------------------
from user_modeling_code_part4 import (
    FeedbackType,
    FeedbackContext,
    FeedbackItem,
    FeedbackCollectionSystem,
    FeedbackPromptingSystem,
    UserRole,
    Permission,
    ROLE_PERMISSIONS,
    UserSession,
    MultiUserManager,
    UserContextIsolation,
    SharedContextManager,
)

__all__ = [
    # Part 1
    "ProfileVersion",
    "UserProfile",
    "UserProfileManager",
    "PreferenceSignal",
    "PreferenceLearningEngine",
    "ContextualPreferenceLearner",
    # Part 2
    "BehaviorPattern",
    "TemporalPatternDetector",
    "SequentialPatternDetector",
    "SemanticPatternDetector",
    "PatternRecognitionOrchestrator",
    "PersonalizationLevel",
    "PersonalizationContext",
    "PersonalizationResult",
    "PersonalizationEngine",
    "ResponsePersonalizer",
    "UIPersonalizer",
    # Part 3
    "AdaptationTrigger",
    "AdaptationPriority",
    "AdaptationAction",
    "UserAdaptationEngine",
    "DataSensitivity",
    "DataCategory",
    "DataField",
    "PrivacySchema",
    "PrivacyManager",
    "ConsentManager",
    # Part 4
    "FeedbackType",
    "FeedbackContext",
    "FeedbackItem",
    "FeedbackCollectionSystem",
    "FeedbackPromptingSystem",
    "UserRole",
    "Permission",
    "ROLE_PERMISSIONS",
    "UserSession",
    "MultiUserManager",
    "UserContextIsolation",
    "SharedContextManager",
]
