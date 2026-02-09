# FORM HANDLING AND USER SIMULATION SYSTEM
## Technical Specification for Windows 10 OpenClaw-Inspired AI Agent

**Version:** 1.0  
**Date:** 2025-01-20  
**Status:** Design Specification  
**Classification:** Technical Architecture Document

---

## TABLE OF CONTENTS

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Form Detection and Analysis](#3-form-detection-and-analysis)
4. [Input Field Identification and Classification](#4-input-field-identification-and-classification)
5. [Form Filling Strategies](#5-form-filling-strategies)
6. [Dropdown and Selection Handling](#6-dropdown-and-selection-handling)
7. [File Upload Automation](#7-file-upload-automation)
8. [Checkbox and Radio Button Handling](#8-checkbox-and-radio-button-handling)
9. [CAPTCHA Detection and Solving](#9-captcha-detection-and-solving)
10. [Human-like Behavior Simulation](#10-human-like-behavior-simulation)
11. [Security and Anti-Detection](#11-security-and-anti-detection)
12. [Implementation Roadmap](#12-implementation-roadmap)

---

## 1. EXECUTIVE SUMMARY

### 1.1 Purpose
This document specifies the architecture and implementation details for a comprehensive form handling and user simulation system designed for a Windows 10-based AI agent framework. The system enables autonomous form detection, intelligent field analysis, human-like interaction patterns, and robust CAPTCHA handling.

### 1.2 Key Capabilities
- **Real-time Form Detection**: DOM analysis and visual recognition
- **Intelligent Field Classification**: ML-powered input type identification
- **Adaptive Form Filling**: Context-aware data population
- **Human Behavior Simulation**: Natural interaction patterns with variable delays
- **CAPTCHA Handling**: Multi-provider solving with fallback strategies
- **Cross-Platform Compatibility**: Browser and native Windows application support

### 1.3 Technical Stack
```
Core Technologies:
├── Python 3.11+ (Primary Language)
├── Playwright/Selenium (Browser Automation)
├── PyAutoGUI/Win32 API (System-level Input)
├── OpenCV (Visual Recognition)
├── TensorFlow/PyTorch (ML Models)
├── 2captcha/Anti-Captcha (CAPTCHA Services)
└── Windows UI Automation (Native App Support)
```

---

## 2. SYSTEM ARCHITECTURE OVERVIEW

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FORM HANDLING ORCHESTRATOR                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Detector   │  │  Classifier  │  │   Filler     │  │   Simulator  │    │
│  │   Module     │→ │   Module     │→ │   Module     │→ │   Module     │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│         ↓                 ↓                 ↓                 ↓             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Form Scanner │  │ Field Typing │  │ Value Gen    │  │ Behavior     │    │
│  │ DOM Parser   │  │ Heuristics   │  │ Strategy     │  │ Engine       │    │
│  │ Visual Recog │  │ ML Models    │  │ Validation   │  │ Delay System │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SUPPORTING SUBSYSTEMS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   CAPTCHA    │  │   Error      │  │   Audit      │  │   Config     │    │
│  │   Handler    │  │   Recovery   │  │   Logger     │  │   Manager    │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Core Components

#### 2.2.1 FormDetector
- **Purpose**: Identify and locate forms on web pages and applications
- **Input**: Page content (HTML/DOM or screenshot)
- **Output**: List of detected forms with metadata
- **Methods**: DOM parsing, visual recognition, hybrid detection

#### 2.2.2 FieldClassifier
- **Purpose**: Categorize input fields by type and purpose
- **Input**: Individual form elements
- **Output**: Field classification with confidence scores
- **Methods**: Attribute analysis, label association, ML inference

#### 2.2.3 FormFiller
- **Purpose**: Populate form fields with appropriate values
- **Input**: Classified fields + context data
- **Output**: Filled form ready for submission
- **Methods**: Strategy-based filling, validation, error handling

#### 2.2.4 BehaviorSimulator
- **Purpose**: Mimic human interaction patterns
- **Input**: Target element coordinates/actions
- **Output**: Simulated human-like interactions
- **Methods**: Mouse movement curves, typing patterns, delay injection

---

## 3. FORM DETECTION AND ANALYSIS

### 3.1 Detection Methods

#### 3.1.1 DOM-Based Detection

```python
class DOMFormDetector:
    """Detects forms by analyzing HTML structure and DOM elements."""
    
    SELECTORS = {
        "forms": [
            "form",
            '[role="form"]',
            ".form",
            ".login-form",
            ".registration-form",
            ".contact-form",
            ".search-form",
            ".checkout-form",
            "[data-form]"
        ],
        "inputs": [
            'input:not([type="hidden"])',
            "textarea",
            "select",
            '[contenteditable="true"]',
            '[role="textbox"]',
            '[role="combobox"]',
            '[role="listbox"]'
        ]
    }
    
    async def detect_forms(self, page) -> List[FormElement]:
        """Primary detection method using DOM analysis."""
        forms = []
        
        # Execute detection script in browser context
        detected = await page.evaluate("""() => {
            const forms = [];
            
            // Find all form containers
            document.querySelectorAll("form, [role='form'], .form").forEach((form, index) => {
                const formData = {
                    id: form.id || `form_${index}`,
                    selector: generateSelector(form),
                    action: form.action || "",
                    method: form.method || "GET",
                    fields: [],
                    bounds: form.getBoundingClientRect()
                };
                
                // Extract all input fields
                form.querySelectorAll("input, textarea, select").forEach(field => {
                    formData.fields.push({
                        tag: field.tagName.toLowerCase(),
                        type: field.type || "text",
                        name: field.name || "",
                        id: field.id || "",
                        placeholder: field.placeholder || "",
                        required: field.required,
                        selector: generateSelector(field),
                        bounds: field.getBoundingClientRect()
                    });
                });
                
                forms.push(formData);
            });
            
            return forms;
        }""")
        
        return [FormElement.from_dict(f) for f in detected]
```

#### 3.1.2 Visual Recognition Detection

```python
class VisualFormDetector:
    """Detects forms using computer vision when DOM analysis fails."""
    
    def __init__(self):
        self.input_field_classifier = self._load_cv_model()
        
    def detect_from_screenshot(self, screenshot_path: str) -> List[VisualFormRegion]:
        """Analyze screenshot to identify form-like regions."""
        import cv2
        import numpy as np
        
        # Load and preprocess image
        image = cv2.imread(screenshot_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect rectangular regions (potential form containers)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        form_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size (forms are typically larger)
            if w > 200 and h > 100:
                roi = image[y:y+h, x:x+w]
                
                # Classify region using ML model
                classification = self.classify_region(roi)
                
                if classification["is_form"]:
                    form_regions.append(VisualFormRegion(
                        x=x, y=y, width=w, height=h,
                        confidence=classification["confidence"],
                        field_regions=self.detect_input_regions(roi, x, y)
                    ))
        
        return form_regions
```

#### 3.1.3 Hybrid Detection Strategy

```python
class HybridFormDetector:
    """Combines DOM and visual detection for maximum accuracy."""
    
    def __init__(self):
        self.dom_detector = DOMFormDetector()
        self.visual_detector = VisualFormDetector()
        
    async def detect(self, page) -> List[UnifiedFormElement]:
        """Execute both detection methods and merge results."""
        # Run DOM detection
        dom_forms = await self.dom_detector.detect_forms(page)
        
        # Take screenshot for visual detection
        screenshot = await page.screenshot()
        visual_forms = self.visual_detector.detect_from_screenshot(screenshot)
        
        # Merge and deduplicate results
        unified_forms = self.merge_detections(dom_forms, visual_forms)
        
        return unified_forms
```

### 3.2 Form Metadata Extraction

```python
@dataclass
class FormMetadata:
    """Comprehensive metadata for detected forms."""
    form_id: str
    form_type: FormType  # LOGIN, REGISTRATION, SEARCH, PAYMENT, etc.
    confidence: float
    field_count: int
    required_fields: int
    has_captcha: bool
    captcha_type: Optional[CaptchaType]
    submit_method: str  # GET, POST, AJAX, etc.
    action_url: Optional[str]
    security_features: List[str]  # CSRF tokens, honeypots, etc.
    estimated_completion_time: timedelta

class FormAnalyzer:
    """Analyzes detected forms to extract metadata and characteristics."""
    
    async def analyze_form(self, form: UnifiedFormElement, page) -> FormMetadata:
        """Perform comprehensive form analysis."""
        # Determine form type
        form_type = self.classify_form_type(form)
        
        # Check for CAPTCHA
        captcha_info = await self.detect_captcha(form, page)
        
        # Analyze security features
        security = await self.analyze_security(form, page)
        
        # Calculate completion estimate
        completion_time = self.estimate_completion(form)
        
        return FormMetadata(
            form_id=form.dom_data.id if form.dom_data else f"visual_{id(form)}",
            form_type=form_type,
            confidence=form.confidence,
            field_count=len(form.fields),
            required_fields=sum(1 for f in form.fields if f.required),
            has_captcha=captcha_info["detected"],
            captcha_type=captcha_info.get("type"),
            submit_method=form.dom_data.method if form.dom_data else "unknown",
            action_url=form.dom_data.action if form.dom_data else None,
            security_features=security,
            estimated_completion_time=completion_time
        )
```

---

## 4. INPUT FIELD IDENTIFICATION AND CLASSIFICATION

### 4.1 Field Type Classification

```python
class FieldClassifier:
    """ML-powered field classification system."""
    
    FIELD_TYPES = {
        # Personal Information
        "first_name": {
            "keywords": ["first", "fname", "firstname"],
            "patterns": [r"^first[_\s]?name$"]
        },
        "last_name": {
            "keywords": ["last", "lname", "lastname", "surname"],
            "patterns": [r"^last[_\s]?name$", r"^surname$"]
        },
        "full_name": {
            "keywords": ["name", "fullname"],
            "patterns": [r"^full[_\s]?name$", r"^name$"]
        },
        "email": {
            "keywords": ["email", "e-mail", "mail"],
            "patterns": [r"email", r"e[-\s]?mail"]
        },
        "phone": {
            "keywords": ["phone", "tel", "mobile", "cell"],
            "patterns": [r"phone", r"telephone", r"mobile"]
        },
        # Address Information
        "address_line1": {
            "keywords": ["address", "street"],
            "patterns": [r"^address[_\s]?1$", r"^street$"]
        },
        "city": {
            "keywords": ["city", "town"],
            "patterns": [r"^city$", r"^town$"]
        },
        "state": {
            "keywords": ["state", "province", "region"],
            "patterns": [r"^state", r"^province"]
        },
        "zip_code": {
            "keywords": ["zip", "postal", "postcode"],
            "patterns": [r"^zip", r"^postal"]
        },
        "country": {
            "keywords": ["country", "nation"],
            "patterns": [r"^country$"]
        },
        # Authentication
        "username": {
            "keywords": ["username", "user", "login"],
            "patterns": [r"^user", r"^login$"]
        },
        "password": {
            "keywords": ["password", "pass", "pwd"],
            "patterns": [r"^pass", r"^pwd$"]
        },
        "confirm_password": {
            "keywords": ["confirm", "verify", "repeat"],
            "patterns": [r"confirm.*pass", r"verify.*pass"]
        },
        # Payment
        "card_number": {
            "keywords": ["card", "ccnum", "cardnum"],
            "patterns": [r"card.*num", r"^cc"]
        },
        "card_expiry": {
            "keywords": ["expiry", "expiration", "exp"],
            "patterns": [r"expir", r"^exp$"]
        },
        "cvv": {
            "keywords": ["cvv", "cvc", "cvv2", "security"],
            "patterns": [r"^cvv", r"^cvc", r"security.*code"]
        },
        # Other
        "date_of_birth": {
            "keywords": ["dob", "birth", "birthdate"],
            "patterns": [r"birth", r"^dob$"]
        },
        "company": {
            "keywords": ["company", "organization", "org"],
            "patterns": [r"^company", r"^org"]
        },
        "job_title": {
            "keywords": ["title", "position", "role"],
            "patterns": [r"job.*title", r"^position$"]
        },
        "website": {
            "keywords": ["website", "url", "site"],
            "patterns": [r"^website", r"^url$"]
        },
    }
    
    def __init__(self):
        self.ml_model = self._load_classification_model()
        self.heuristic_weights = self._load_heuristic_config()
    
    def classify_field(self, field: FieldElement, context: FormContext) -> FieldClassification:
        """Classify a field using multiple methods and combine results."""
        # Method 1: Heuristic classification
        heuristic_result = self.heuristic_classify(field)
        
        # Method 2: ML classification
        ml_result = self.ml_classify(field, context)
        
        # Method 3: Context-based classification
        context_result = self.context_classify(field, context)
        
        # Combine results with weighted voting
        combined = self.combine_classifications(
            heuristic_result, ml_result, context_result
        )
        
        return combined
```

### 4.2 Field Relationship Analysis

```python
class FieldRelationshipAnalyzer:
    """Analyzes relationships between form fields."""
    
    def analyze_relationships(self, fields: List[ClassifiedField]) -> FieldRelationshipGraph:
        """Build a graph of field relationships."""
        graph = FieldRelationshipGraph()
        
        for i, field1 in enumerate(fields):
            for j, field2 in enumerate(fields):
                if i >= j:
                    continue
                
                # Check for grouping
                if self.are_grouped(field1, field2):
                    graph.add_edge(field1, field2, RelationshipType.GROUPED)
                
                # Check for dependency
                if self.are_dependent(field1, field2):
                    graph.add_edge(field1, field2, RelationshipType.DEPENDENT)
                
                # Check for confirmation relationship
                if self.is_confirmation_pair(field1, field2):
                    graph.add_edge(field1, field2, RelationshipType.CONFIRMATION)
        
        return graph
    
    def are_grouped(self, field1: ClassifiedField, field2: ClassifiedField) -> bool:
        """Check if fields belong to the same logical group."""
        # Check DOM parent
        if field1.parent_element and field2.parent_element:
            if field1.parent_element == field2.parent_element:
                return True
        
        # Check fieldset
        if field1.fieldset and field2.fieldset:
            if field1.fieldset == field2.fieldset:
                return True
        
        # Check proximity
        if field1.bounds and field2.bounds:
            vertical_distance = abs(field1.bounds.y - field2.bounds.y)
            if vertical_distance < 50:  # Same row
                return True
        
        return False
    
    def is_confirmation_pair(self, field1: ClassifiedField, field2: ClassifiedField) -> bool:
        """Detect password/email confirmation pairs."""
        confirm_keywords = ["confirm", "verify", "repeat", "retype", "again"]
        
        name1 = (field1.name or "").lower()
        name2 = (field2.name or "").lower()
        
        # One should have confirm keyword
        has_confirm = any(kw in name1 for kw in confirm_keywords) or \
                      any(kw in name2 for kw in confirm_keywords)
        
        # Base types should match
        base_type1 = field1.classification.field_type.replace("confirm_", "")
        base_type2 = field2.classification.field_type.replace("confirm_", "")
        
        return has_confirm and base_type1 == base_type2
```

---

## 5. FORM FILLING STRATEGIES

### 5.1 Strategy Pattern Implementation

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class FillingStrategy(ABC):
    """Abstract base class for form filling strategies."""
    
    @abstractmethod
    async def fill_field(self, field: ClassifiedField, context: FillContext) -> FillResult:
        """Fill a single field."""
        pass
    
    @abstractmethod
    def can_handle(self, field: ClassifiedField) -> bool:
        """Check if this strategy can handle the field."""
        pass

class TextFieldStrategy(FillingStrategy):
    """Strategy for filling text input fields."""
    
    VALUE_GENERATORS = {
        "first_name": lambda ctx: ctx.profile.first_name or fake.first_name(),
        "last_name": lambda ctx: ctx.profile.last_name or fake.last_name(),
        "full_name": lambda ctx: ctx.profile.full_name or fake.name(),
        "email": lambda ctx: ctx.profile.email or fake.email(),
        "phone": lambda ctx: ctx.profile.phone or fake.phone_number(),
        "username": lambda ctx: ctx.profile.username or fake.user_name(),
        "company": lambda ctx: ctx.profile.company or fake.company(),
        "job_title": lambda ctx: ctx.profile.job_title or fake.job(),
        "website": lambda ctx: ctx.profile.website or fake.url(),
    }
    
    def can_handle(self, field: ClassifiedField) -> bool:
        return field.element.type in ["text", "email", "tel", "url", "password"]
    
    async def fill_field(self, field: ClassifiedField, context: FillContext) -> FillResult:
        field_type = field.classification.field_type
        
        # Get value generator
        generator = self.VALUE_GENERATORS.get(field_type)
        
        if generator:
            value = generator(context)
        else:
            # Fallback to generic generation
            value = self.generate_generic_value(field, context)
        
        # Apply transformations
        value = self.transform_value(value, field)
        
        # Fill the field
        await self.input_value(field, value, context)
        
        return FillResult(
            success=True,
            field_id=field.element.id,
            value=value,
            strategy="text"
        )
    
    async def input_value(self, field: ClassifiedField, value: str, context: FillContext):
        """Input value with human-like behavior."""
        page = context.page
        selector = field.element.selector
        
        # Clear existing value
        await page.click(selector)
        await page.keyboard.press("Control+a")
        await page.keyboard.press("Delete")
        
        # Type with variable delays
        for char in value:
            await page.keyboard.press(char)
            
            # Random delay between keystrokes (50-150ms)
            delay = random.randint(50, 150)
            
            # Add occasional longer pauses (typing hesitation)
            if random.random() < 0.05:  # 5% chance
                delay += random.randint(200, 500)
            
            await asyncio.sleep(delay / 1000)

class PasswordStrategy(FillingStrategy):
    """Strategy for password fields with security considerations."""
    
    def can_handle(self, field: ClassifiedField) -> bool:
        return field.classification.field_type in ["password", "confirm_password"]
    
    async def fill_field(self, field: ClassifiedField, context: FillContext) -> FillResult:
        # Check if this is a confirmation field
        if field.classification.field_type == "confirm_password":
            # Find the original password field
            original = self.find_password_pair(field, context)
            if original:
                value = original.filled_value
            else:
                value = context.profile.password or self.generate_password()
        else:
            value = context.profile.password or self.generate_password()
        
        # Store for confirmation fields
        context.password_cache[field.element.form_id] = value
        
        await self.input_value(field, value, context)
        
        return FillResult(
            success=True,
            field_id=field.element.id,
            value="[REDACTED]",
            strategy="password"
        )
    
    def generate_password(self, length: int = 16) -> str:
        """Generate a secure password meeting common requirements."""
        import secrets
        import string
        
        # Ensure at least one of each required character type
        password = [
            secrets.choice(string.ascii_uppercase),
            secrets.choice(string.ascii_lowercase),
            secrets.choice(string.digits),
            secrets.choice("!@#$%^&*")
        ]
        
        # Fill remaining length
        all_chars = string.ascii_letters + string.digits + "!@#$%^&*"
        password.extend(secrets.choice(all_chars) for _ in range(length - 4))
        
        # Shuffle
        secrets.SystemRandom().shuffle(password)
        
        return "".join(password)
```

### 5.2 Form Filling Orchestrator

```python
class FormFillingOrchestrator:
    """Orchestrates the complete form filling process."""
    
    STRATEGIES = [
        TextFieldStrategy(),
        PasswordStrategy(),
        DateFieldStrategy(),
        DropdownStrategy(),
        CheckboxStrategy(),
        RadioStrategy(),
        FileUploadStrategy(),
        TextAreaStrategy(),
    ]
    
    def __init__(self):
        self.classifier = FieldClassifier()
        self.relationship_analyzer = FieldRelationshipAnalyzer()
        self.validator = FormValidator()
        self.behavior_simulator = BehaviorSimulator()
    
    async def fill_form(self, form: UnifiedFormElement, context: FillContext) -> FormFillResult:
        """Main entry point for form filling."""
        results = []
        
        # Phase 1: Classify all fields
        classified_fields = []
        for field in form.fields:
            classification = self.classifier.classify_field(field, context)
            classified_fields.append(ClassifiedField(
                element=field,
                classification=classification
            ))
        
        # Phase 2: Analyze relationships
        relationship_graph = self.relationship_analyzer.analyze_relationships(classified_fields)
        
        # Phase 3: Determine filling order
        filling_order = self.determine_filling_order(classified_fields, relationship_graph)
        
        # Phase 4: Fill fields in order
        for field in filling_order:
            strategy = self.select_strategy(field)
            
            if not strategy:
                results.append(FillResult(
                    success=False,
                    field_id=field.element.id,
                    error="No suitable strategy found"
                ))
                continue
            
            # Simulate human behavior before filling
            await self.behavior_simulator.prepare_for_interaction(field, context)
            
            # Fill the field
            try:
                result = await strategy.fill_field(field, context)
                results.append(result)
                
                # Store filled value for relationship handling
                field.filled_value = result.value
                
            except Exception as e:
                results.append(FillResult(
                    success=False,
                    field_id=field.element.id,
                    error=str(e)
                ))
            
            # Random delay between fields
            await asyncio.sleep(random.uniform(0.5, 2.0))
        
        # Phase 5: Validate filled form
        validation = await self.validator.validate(form, context)
        
        return FormFillResult(
            form_id=form.form_id,
            field_results=results,
            validation=validation,
            success=all(r.success for r in results) and validation.valid
        )
    
    def select_strategy(self, field: ClassifiedField) -> Optional[FillingStrategy]:
        """Select the appropriate strategy for a field."""
        for strategy in self.STRATEGIES:
            if strategy.can_handle(field):
                return strategy
        return None
    
    def determine_filling_order(self, fields: List[ClassifiedField], 
                                 graph: FieldRelationshipGraph) -> List[ClassifiedField]:
        """Determine optimal field filling order."""
        # Topological sort for dependency handling
        ordered = []
        visited = set()
        
        def visit(field):
            if field in visited:
                return
            visited.add(field)
            
            # Visit dependencies first
            for dependency in graph.get_dependencies(field):
                visit(dependency)
            
            ordered.append(field)
        
        for field in fields:
            visit(field)
        
        return ordered
```

---

## 6. DROPDOWN AND SELECTION HANDLING

### 6.1 Dropdown Strategy

```python
class DropdownStrategy(FillingStrategy):
    """Strategy for handling select dropdowns and custom dropdowns."""
    
    def can_handle(self, field: ClassifiedField) -> bool:
        return field.element.tag == "select" or \
               field.element.role in ["combobox", "listbox"]
    
    async def fill_field(self, field: ClassifiedField, context: FillContext) -> FillResult:
        # Get available options
        options = await self.get_options(field, context)
        
        # Determine target value
        target_value = self.determine_target_value(field, options, context)
        
        # Select the option
        await self.select_option(field, target_value, context)
        
        return FillResult(
            success=True,
            field_id=field.element.id,
            value=target_value
        )
    
    async def get_options(self, field: ClassifiedField, context: FillContext) -> List[Option]:
        """Extract all available options from dropdown."""
        page = context.page
        selector = field.element.selector
        
        if field.element.tag == "select":
            # Native select element
            options = await page.evaluate(f"""(selector) => {{
                const select = document.querySelector(selector);
                return Array.from(select.options).map(opt => ({{
                    value: opt.value,
                    text: opt.text.trim(),
                    selected: opt.selected
                }}));
            }}""", selector)
        else:
            # Custom dropdown (React, Vue, etc.)
            options = await self.extract_custom_options(field, context)
        
        return [Option(o["value"], o["text"], o["selected"]) for o in options]
    
    async def extract_custom_options(self, field: ClassifiedField, context: FillContext) -> List[Dict]:
        """Extract options from custom dropdown implementations."""
        page = context.page
        
        # Click to open dropdown
        await page.click(field.element.selector)
        await asyncio.sleep(0.3)  # Wait for animation
        
        # Try common option selectors
        option_selectors = [
            '[role="option"]',
            ".dropdown-item",
            ".select-option",
            ".menu-item",
            "li"
        ]
        
        for opt_selector in option_selectors:
            options = await page.query_selector_all(opt_selector)
            if options:
                result = []
                for opt in options:
                    text = await opt.text_content()
                    value = await opt.get_attribute("data-value") or text
                    result.append({"value": value, "text": text.strip(), "selected": False})
                return result
        
        return []
    
    def determine_target_value(self, field: ClassifiedField, 
                                options: List[Option], 
                                context: FillContext) -> str:
        """Determine which option to select based on field type and context."""
        field_type = field.classification.field_type
        
        # Country selection
        if field_type == "country":
            target = context.profile.country or "United States"
            return self.find_best_match(target, options)
        
        # State selection
        if field_type == "state":
            target = context.profile.state
            return self.find_best_match(target, options)
        
        # Generic selection - avoid placeholder options
        valid_options = [o for o in options if not self.is_placeholder(o)]
        
        if valid_options:
            # Prefer options that match context
            for option in valid_options:
                if self.matches_context(option, field_type, context):
                    return option.value
            
            # Random selection from valid options
            return random.choice(valid_options).value
        
        # Fallback to first non-empty option
        return options[0].value if options else ""
    
    def find_best_match(self, target: str, options: List[Option]) -> str:
        """Find the best matching option for a target value."""
        if not target:
            return options[0].value if options else ""
        
        target_lower = target.lower()
        
        # Exact match
        for opt in options:
            if opt.text.lower() == target_lower or opt.value.lower() == target_lower:
                return opt.value
        
        # Partial match
        for opt in options:
            if target_lower in opt.text.lower() or target_lower in opt.value.lower():
                return opt.value
        
        # Fuzzy match using Levenshtein distance
        best_match = None
        best_score = float("inf")
        
        for opt in options:
            score = levenshtein_distance(target_lower, opt.text.lower())
            if score < best_score:
                best_score = score
                best_match = opt
        
        return best_match.value if best_match else options[0].value
    
    async def select_option(self, field: ClassifiedField, value: str, context: FillContext):
        """Select an option with human-like interaction."""
        page = context.page
        selector = field.element.selector
        
        if field.element.tag == "select":
            # Native select - use Playwright's select_option
            await page.select_option(selector, value)
        else:
            # Custom dropdown
            await self.select_custom_option(field, value, context)
    
    async def select_custom_option(self, field: ClassifiedField, value: str, context: FillContext):
        """Select option from custom dropdown implementation."""
        page = context.page
        
        # Click to open
        await page.click(field.element.selector)
        await asyncio.sleep(0.3)
        
        # Find and click the option
        option_selectors = [
            f'[role="option"]:has-text("{value}")',
            f'.dropdown-item:has-text("{value}")',
            f'[data-value="{value}"]'
        ]
        
        for opt_selector in option_selectors:
            try:
                await page.click(opt_selector, timeout=1000)
                return
            except:
                continue
        
        # Fallback: try to find by partial text
        options = await page.query_selector_all('[role="option"], .dropdown-item')
        for opt in options:
            text = await opt.text_content()
            if value.lower() in (text or "").lower():
                await opt.click()
                return
```

### 6.2 Multi-Select Handling

```python
class MultiSelectStrategy(FillingStrategy):
    """Strategy for multi-select dropdowns and checkboxes."""
    
    def can_handle(self, field: ClassifiedField) -> bool:
        return field.element.tag == "select" and field.element.multiple
    
    async def fill_field(self, field: ClassifiedField, context: FillContext) -> FillResult:
        options = await self.get_options(field, context)
        
        # Determine how many to select
        num_to_select = self.determine_selection_count(field, options)
        
        # Select values
        selected = []
        for _ in range(num_to_select):
            target = self.determine_target_value(field, options, context)
            await self.select_option(field, target, context)
            selected.append(target)
            
            # Remove selected from available options
            options = [o for o in options if o.value != target]
            
            # Small delay between selections
            await asyncio.sleep(0.2)
        
        return FillResult(
            success=True,
            field_id=field.element.id,
            value=selected
        )
    
    def determine_selection_count(self, field: ClassifiedField, options: List[Option]) -> int:
        """Determine how many options to select."""
        # Check for min/max constraints
        min_required = field.element.min_selections or 1
        max_allowed = field.element.max_selections or len(options)
        
        # Random selection within constraints
        return random.randint(min_required, min(max_allowed, len(options)))
```

---

## 7. FILE UPLOAD AUTOMATION

### 7.1 File Upload Strategy

```python
class FileUploadStrategy(FillingStrategy):
    """Strategy for handling file upload fields."""
    
    def can_handle(self, field: ClassifiedField) -> bool:
        return field.element.type == "file"
    
    async def fill_field(self, field: ClassifiedField, context: FillContext) -> FillResult:
        # Determine file requirements
        requirements = self.analyze_requirements(field)
        
        # Generate or select appropriate file
        file_path = await self.prepare_file(requirements, context)
        
        # Upload the file
        await self.upload_file(field, file_path, context)
        
        # Verify upload
        verification = await self.verify_upload(field, context)
        
        return FillResult(
            success=verification.success,
            field_id=field.element.id,
            value=file_path,
            metadata={"file_size": os.path.getsize(file_path), "verification": verification}
        )
    
    def analyze_requirements(self, field: ClassifiedField) -> FileRequirements:
        """Analyze file upload requirements from field attributes."""
        accept = field.element.accept or ""
        
        # Parse accept attribute
        allowed_types = []
        if accept:
            allowed_types = [t.strip() for t in accept.split(",")]
        
        # Check for max size (often in data attributes)
        max_size = field.element.max_file_size
        
        # Check for multiple files
        multiple = field.element.multiple
        
        return FileRequirements(
            allowed_types=allowed_types or ["*/*"],
            max_size=max_size or 10 * 1024 * 1024,  # Default 10MB
            multiple=multiple,
            min_files=1,
            max_files=field.element.max_files or 1
        )
    
    async def prepare_file(self, requirements: FileRequirements, 
                          context: FillContext) -> str:
        """Generate or select an appropriate file for upload."""
        # Determine file type
        file_type = self.determine_file_type(requirements.allowed_types)
        
        # Check if we have a suitable file in cache
        cached = self.find_cached_file(file_type, requirements)
        if cached:
            return cached
        
        # Generate new file
        if file_type == "image":
            return await self.generate_image_file(requirements)
        elif file_type == "document":
            return await self.generate_document_file(requirements)
        elif file_type == "pdf":
            return await self.generate_pdf_file(requirements)
        else:
            return await self.generate_generic_file(requirements)
    
    async def generate_image_file(self, requirements: FileRequirements) -> str:
        """Generate a sample image file."""
        from PIL import Image
        import io
        
        # Create a simple test image
        width, height = 800, 600
        image = Image.new("RGB", (width, height), color=(random.randint(0, 255), 
                                                          random.randint(0, 255), 
                                                          random.randint(0, 255)))
        
        # Add some random shapes
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        for _ in range(10):
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            draw.rectangle([x1, y1, x2, y2], 
                          fill=(random.randint(0, 255), 
                                random.randint(0, 255), 
                                random.randint(0, 255)))
        
        # Save to temp file
        temp_path = f"/tmp/upload_{uuid.uuid4()}.png"
        image.save(temp_path, "PNG")
        
        return temp_path
    
    async def upload_file(self, field: ClassifiedField, file_path: str, context: FillContext):
        """Upload file to the input element."""
        page = context.page
        selector = field.element.selector
        
        # Use Playwright's file upload
        await page.set_input_files(selector, file_path)
        
        # Wait for any upload progress indicators
        await asyncio.sleep(1)
    
    async def verify_upload(self, field: ClassifiedField, context: FillContext) -> UploadVerification:
        """Verify that the file was uploaded successfully."""
        page = context.page
        
        # Check for success indicators
        success_selectors = [
            ".upload-success",
            ".file-uploaded",
            '[data-upload-status="success"]'
        ]
        
        for selector in success_selectors:
            element = await page.query_selector(selector)
            if element:
                return UploadVerification(success=True, method="success_indicator")
        
        # Check for error indicators
        error_selectors = [
            ".upload-error",
            ".file-error",
            '[data-upload-status="error"]'
        ]
        
        for selector in error_selectors:
            element = await page.query_selector(selector)
            if element:
                error_text = await element.text_content()
                return UploadVerification(success=False, error=error_text)
        
        # Check if file input still has value
        has_file = await page.evaluate(f"""(selector) => {{
                const input = document.querySelector(selector);
                return input && input.files && input.files.length > 0;
            }}""", field.element.selector)
        
        return UploadVerification(success=has_file, method="file_check")
```

### 7.2 Drag-and-Drop Upload

```python
class DragDropUploadHandler:
    """Handle drag-and-drop file uploads."""
    
    async def upload_via_drag_drop(self, drop_zone_selector: str, 
                                    file_path: str, 
                                    context: FillContext):
        """Simulate drag-and-drop file upload."""
        page = context.page
        
        # Read file data
        with open(file_path, "rb") as f:
            file_data = f.read()
        
        file_name = os.path.basename(file_path)
        mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        
        # Create DataTransfer object with file
        await page.evaluate(f"""
            async (selector, fileData, fileName, mimeType) => {{
                const dropZone = document.querySelector(selector);
                
                // Create File object
                const blob = new Blob([new Uint8Array({list(file_data)})], {{ type: mimeType }});
                const file = new File([blob], fileName, {{ type: mimeType }});
                
                // Create DataTransfer
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                
                // Dispatch dragenter event
                const dragEnterEvent = new DragEvent("dragenter", {{
                    bubbles: true,
                    cancelable: true,
                    dataTransfer: dataTransfer
                }});
                dropZone.dispatchEvent(dragEnterEvent);
                
                // Dispatch dragover event
                const dragOverEvent = new DragEvent("dragover", {{
                    bubbles: true,
                    cancelable: true,
                    dataTransfer: dataTransfer
                }});
                dropZone.dispatchEvent(dragOverEvent);
                
                // Small delay
                await new Promise(r => setTimeout(r, 100));
                
                // Dispatch drop event
                const dropEvent = new DragEvent("drop", {{
                    bubbles: true,
                    cancelable: true,
                    dataTransfer: dataTransfer
                }});
                dropZone.dispatchEvent(dropEvent);
            }}
        """, drop_zone_selector, list(file_data), file_name, mime_type)
```

---

## 8. CHECKBOX AND RADIO BUTTON HANDLING

### 8.1 Checkbox Strategy

```python
class CheckboxStrategy(FillingStrategy):
    """Strategy for handling checkbox inputs."""
    
    def can_handle(self, field: ClassifiedField) -> bool:
        return field.element.type == "checkbox"
    
    async def fill_field(self, field: ClassifiedField, context: FillContext) -> FillResult:
        # Determine if checkbox should be checked
        should_check = self.determine_check_state(field, context)
        
        # Get current state
        current_state = await self.get_current_state(field, context)
        
        # Toggle if needed
        if should_check != current_state:
            await self.toggle_checkbox(field, context)
        
        return FillResult(
            success=True,
            field_id=field.element.id,
            value=should_check
        )
    
    def determine_check_state(self, field: ClassifiedField, context: FillContext) -> bool:
        """Determine whether checkbox should be checked."""
        # Check if required
        if field.element.required:
            return True
        
        # Check label text for indicators
        label = (field.label_text or "").lower()
        
        # Terms and conditions - usually required
        if any(kw in label for kw in ["terms", "agree", "accept", "consent"]):
            return True
        
        # Marketing/optional - usually unchecked
        if any(kw in label for kw in ["marketing", "newsletter", "promotional", "opt-in"]):
            return random.random() < 0.3  # 30% chance to opt-in
        
        # Remember me - random
        if any(kw in label for kw in ["remember", "keep", "save"]):
            return random.random() < 0.5
        
        # Default: random with slight bias toward unchecked
        return random.random() < 0.3
    
    async def get_current_state(self, field: ClassifiedField, context: FillContext) -> bool:
        """Get current checked state of checkbox."""
        page = context.page
        return await page.evaluate(f"""(selector) => {{
                const checkbox = document.querySelector(selector);
                return checkbox ? checkbox.checked : false;
            }}""", field.element.selector)
    
    async def toggle_checkbox(self, field: ClassifiedField, context: FillContext):
        """Toggle checkbox with human-like click."""
        page = context.page
        
        # Get checkbox position
        bounds = await self.get_element_bounds(field, context)
        
        # Calculate click position (center of checkbox)
        click_x = bounds["x"] + bounds["width"] / 2
        click_y = bounds["y"] + bounds["height"] / 2
        
        # Add small random offset (human imprecision)
        click_x += random.randint(-2, 2)
        click_y += random.randint(-2, 2)
        
        # Move mouse to checkbox with curve
        await context.behavior_simulator.move_mouse(click_x, click_y)
        
        # Small pause before click
        await asyncio.sleep(random.uniform(0.1, 0.3))
        
        # Click
        await page.mouse.click(click_x, click_y)
```

### 8.2 Radio Button Strategy

```python
class RadioStrategy(FillingStrategy):
    """Strategy for handling radio button groups."""
    
    def can_handle(self, field: ClassifiedField) -> bool:
        return field.element.type == "radio"
    
    async def fill_field(self, field: ClassifiedField, context: FillContext) -> FillResult:
        # Get all radio buttons in the group
        radio_group = await self.get_radio_group(field, context)
        
        # Select appropriate option
        selected = await self.select_radio_option(radio_group, field, context)
        
        return FillResult(
            success=selected is not None,
            field_id=field.element.id,
            value=selected.value if selected else None
        )
    
    async def get_radio_group(self, field: ClassifiedField, context: FillContext) -> List[RadioOption]:
        """Get all radio buttons in the same group."""
        page = context.page
        group_name = field.element.name
        
        radios = await page.evaluate(f"""(name) => {{
                const radios = document.querySelectorAll(`input[type="radio"][name="${{name}}"]`);
                return Array.from(radios).map((r, i) => ({{
                    index: i,
                    value: r.value,
                    id: r.id,
                    label: document.querySelector(`label[for="${{r.id}}"]`)?.textContent?.trim() || 
                           r.closest("label")?.textContent?.trim() || r.value
                }}));
            }}""", group_name)
        
        return [RadioOption(**r) for r in radios]
    
    async def select_radio_option(self, radio_group: List[RadioOption], 
                                   field: ClassifiedField, 
                                   context: FillContext) -> Optional[RadioOption]:
        """Select the most appropriate radio option."""
        if not radio_group:
            return None
        
        # Check for preferred values based on field type
        field_type = field.classification.field_type
        
        if field_type == "gender":
            preferred = ["male", "female", "other", "prefer_not_to_say"]
        elif field_type == "title":
            preferred = ["mr", "mrs", "ms", "dr"]
        else:
            preferred = []
        
        # Try to match preferred values
        for pref in preferred:
            for option in radio_group:
                if pref.lower() in option.value.lower() or pref.lower() in option.label.lower():
                    await self.click_radio(option, context)
                    return option
        
        # Default: select first non-empty option or random
        valid_options = [r for r in radio_group if r.value]
        if valid_options:
            selected = random.choice(valid_options)
            await self.click_radio(selected, context)
            return selected
        
        return None
    
    async def click_radio(self, option: RadioOption, context: FillContext):
        """Click a radio button with human-like behavior."""
        page = context.page
        
        # Try to find by ID first
        if option.id:
            selector = f'input[type="radio"][id="{option.id}"]'
        else:
            selector = f'input[type="radio"][value="{option.value}"]'
        
        # Get position
        bounds = await page.evaluate(f"""(selector) => {{
                const el = document.querySelector(selector);
                const rect = el.getBoundingClientRect();
                return {{ x: rect.x, y: rect.y, width: rect.width, height: rect.height }};
            }}""", selector)
        
        # Click with human-like movement
        click_x = bounds["x"] + bounds["width"] / 2 + random.randint(-2, 2)
        click_y = bounds["y"] + bounds["height"] / 2 + random.randint(-2, 2)
        
        await context.behavior_simulator.move_mouse(click_x, click_y)
        await asyncio.sleep(random.uniform(0.1, 0.3))
        await page.mouse.click(click_x, click_y)
```

---

## 9. CAPTCHA DETECTION AND SOLVING

### 9.1 CAPTCHA Detection

```python
class CaptchaDetector:
    """Detects and classifies CAPTCHA challenges."""
    
    CAPTCHA_INDICATORS = {
        "recaptcha_v2": {
            "selectors": [
                ".g-recaptcha",
                "[data-sitekey]",
                'iframe[src*="recaptcha"]',
                "#recaptcha"
            ],
            "text_patterns": [
                r"i\'m not a robot",
                r"recaptcha",
                r"security check"
            ]
        },
        "recaptcha_v3": {
            "selectors": [
                'script[src*="recaptcha/api.js"]',
                "[data-recaptcha-action]"
            ],
            "invisible": True
        },
        "hcaptcha": {
            "selectors": [
                ".h-captcha",
                "[data-sitekey][data-hcaptcha]",
                'iframe[src*="hcaptcha"]'
            ],
            "text_patterns": [
                r"hcaptcha",
                r"prove you are human"
            ]
        },
        "image_captcha": {
            "selectors": [
                'img[src*="captcha"]',
                'img[id*="captcha"]',
                ".captcha-image"
            ],
            "visual_indicators": ["distorted_text", "grid_challenge"]
        },
        "text_captcha": {
            "selectors": [
                ".captcha-question",
                '[name*="captcha"]'
            ],
            "text_patterns": [
                r"what is \d+ [\+\-\*] \d+",
                r"solve this",
                r"enter the code"
            ]
        },
        "funcaptcha": {
            "selectors": [
                'iframe[src*="funcaptcha"]',
                "[data-funcaptcha]"
            ]
        }
    }
    
    async def detect_captcha(self, page) -> CaptchaDetectionResult:
        """Detect CAPTCHA on the current page."""
        for captcha_type, indicators in self.CAPTCHA_INDICATORS.items():
            detected = await self.check_indicators(page, indicators)
            if detected:
                return CaptchaDetectionResult(
                    detected=True,
                    type=captcha_type,
                    confidence=detected["confidence"],
                    element=detected.get("element"),
                    metadata=detected.get("metadata", {})
                )
        
        return CaptchaDetectionResult(detected=False)
    
    async def check_indicators(self, page, indicators: Dict) -> Optional[Dict]:
        """Check for CAPTCHA indicators on the page."""
        # Check selectors
        for selector in indicators.get("selectors", []):
            element = await page.query_selector(selector)
            if element:
                return {
                    "confidence": 0.9,
                    "element": element,
                    "selector": selector
                }
        
        # Check text patterns
        page_text = await page.content()
        for pattern in indicators.get("text_patterns", []):
            if re.search(pattern, page_text, re.IGNORECASE):
                return {
                    "confidence": 0.8,
                    "pattern": pattern
                }
        
        return None
```

### 9.2 CAPTCHA Solving Integration

```python
class CaptchaSolver:
    """Integrates with CAPTCHA solving services."""
    
    SOLVERS = {
        "2captcha": TwoCaptchaProvider(),
        "anticaptcha": AntiCaptchaProvider(),
        "capsolver": CapSolverProvider(),
    }
    
    def __init__(self, config: SolverConfig):
        self.config = config
        self.primary_solver = self.SOLVERS.get(config.primary_provider)
        self.fallback_solvers = [self.SOLVERS[p] for p in config.fallback_providers]
    
    async def solve(self, captcha: CaptchaDetectionResult, page) -> CaptchaSolution:
        """Solve detected CAPTCHA using configured providers."""
        # Try primary solver
        try:
            solution = await self.primary_solver.solve(captcha, page)
            if solution.success:
                return solution
        except Exception as e:
            logger.warning(f"Primary solver failed: {e}")
        
        # Try fallback solvers
        for solver in self.fallback_solvers:
            try:
                solution = await solver.solve(captcha, page)
                if solution.success:
                    return solution
            except Exception as e:
                logger.warning(f"Fallback solver failed: {e}")
        
        return CaptchaSolution(success=False, error="All solvers failed")

class TwoCaptchaProvider:
    """2captcha.com integration."""
    
    API_URL = "http://2captcha.com"
    
    async def solve(self, captcha: CaptchaDetectionResult, page) -> CaptchaSolution:
        if captcha.type == "recaptcha_v2":
            return await self.solve_recaptcha_v2(captcha, page)
        elif captcha.type == "hcaptcha":
            return await self.solve_hcaptcha(captcha, page)
        elif captcha.type == "image_captcha":
            return await self.solve_image_captcha(captcha, page)
        else:
            return CaptchaSolution(success=False, error=f"Unsupported CAPTCHA type: {captcha.type}")
    
    async def solve_recaptcha_v2(self, captcha: CaptchaDetectionResult, page) -> CaptchaSolution:
        """Solve reCAPTCHA v2 challenge."""
        # Get sitekey
        sitekey = await page.evaluate("""() => {{
                return document.querySelector(".g-recaptcha")?.dataset.sitekey ||
                       document.querySelector("[data-sitekey]")?.dataset.sitekey;
            }}""")
        
        page_url = page.url
        
        # Submit to 2captcha
        async with aiohttp.ClientSession() as session:
            # Request solving
            submit_url = f"{self.API_URL}/in.php"
            data = {
                "key": self.api_key,
                "method": "userrecaptcha",
                "googlekey": sitekey,
                "pageurl": page_url,
                "json": 1
            }
            
            async with session.post(submit_url, data=data) as resp:
                result = await resp.json()
                captcha_id = result.get("request")
            
            # Poll for result
            result_url = f"{self.API_URL}/res.php"
            for _ in range(60):  # Max 60 attempts (2 minutes)
                await asyncio.sleep(2)
                
                params = {
                    "key": self.api_key,
                    "action": "get",
                    "id": captcha_id,
                    "json": 1
                }
                
                async with session.get(result_url, params=params) as resp:
                    result = await resp.json()
                    
                    if result.get("status") == 1:
                        token = result.get("request")
                        
                        # Inject token into page
                        await page.evaluate(f"""(token) => {{
                                document.getElementById("g-recaptcha-response").innerHTML = token;
                                if (typeof grecaptcha !== "undefined") {{
                                    grecaptcha.getResponse = () => token;
                                }}
                            }}""", token)
                        
                        return CaptchaSolution(success=True, token=token)
                    
                    elif result.get("request") != "CAPCHA_NOT_READY":
                        return CaptchaSolution(success=False, error=result.get("request"))
        
        return CaptchaSolution(success=False, error="Timeout waiting for solution")
    
    async def solve_image_captcha(self, captcha: CaptchaDetectionResult, page) -> CaptchaSolution:
        """Solve image-based CAPTCHA."""
        # Capture CAPTCHA image
        captcha_element = captcha.element
        screenshot = await captcha_element.screenshot()
        
        # Encode as base64
        image_base64 = base64.b64encode(screenshot).decode()
        
        # Submit to 2captcha
        async with aiohttp.ClientSession() as session:
            submit_url = f"{self.API_URL}/in.php"
            data = {
                "key": self.api_key,
                "method": "base64",
                "body": image_base64,
                "json": 1
            }
            
            async with session.post(submit_url, data=data) as resp:
                result = await resp.json()
                captcha_id = result.get("request")
            
            # Poll for result
            result_url = f"{self.API_URL}/res.php"
            for _ in range(30):
                await asyncio.sleep(2)
                
                params = {
                    "key": self.api_key,
                    "action": "get",
                    "id": captcha_id,
                    "json": 1
                }
                
                async with session.get(result_url, params=params) as resp:
                    result = await resp.json()
                    
                    if result.get("status") == 1:
                        text = result.get("request")
                        
                        # Fill the CAPTCHA field
                        input_selector = 'input[name*="captcha"], .captcha-input'
                        await page.fill(input_selector, text)
                        
                        return CaptchaSolution(success=True, text=text)
        
        return CaptchaSolution(success=False, error="Timeout waiting for solution")
```

---

## 10. HUMAN-LIKE BEHAVIOR SIMULATION

### 10.1 Mouse Movement Simulation

```python
class MouseSimulator:
    """Simulates human-like mouse movements."""
    
    def __init__(self):
        self.current_position = (0, 0)
        self.movement_history = []
    
    async def move_to(self, page, target_x: int, target_y: int, 
                      duration: Optional[float] = None):
        """Move mouse to target position with human-like curve."""
        start_x, start_y = self.current_position
        
        if duration is None:
            # Calculate duration based on distance
            distance = math.sqrt((target_x - start_x)**2 + (target_y - start_y)**2)
            duration = self.calculate_duration(distance)
        
        # Generate bezier curve points
        points = self.generate_bezier_curve(
            start_x, start_y,
            target_x, target_y,
            num_points=max(int(duration * 60), 10)  # 60fps
        )
        
        # Execute movement
        start_time = time.time()
        for point in points:
            # Calculate delay for this point
            elapsed = time.time() - start_time
            expected_time = (points.index(point) / len(points)) * duration
            
            if expected_time > elapsed:
                await asyncio.sleep(expected_time - elapsed)
            
            # Add micro-jitter
            jitter_x = random.gauss(0, 0.5)
            jitter_y = random.gauss(0, 0.5)
            
            await page.mouse.move(point[0] + jitter_x, point[1] + jitter_y)
            self.current_position = (point[0], point[1])
            self.movement_history.append((point[0], point[1], time.time()))
    
    def generate_bezier_curve(self, x1: float, y1: float, 
                               x2: float, y2: float, 
                               num_points: int) -> List[Tuple[float, float]]:
        """Generate points along a bezier curve for natural movement."""
        # Calculate control points for curve
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # Add randomness to control point
        control_x = mid_x + random.gauss(0, abs(x2 - x1) * 0.2)
        control_y = mid_y + random.gauss(0, abs(y2 - y1) * 0.2)
        
        points = []
        for t in np.linspace(0, 1, num_points):
            # Quadratic bezier curve
            x = (1-t)**2 * x1 + 2*(1-t)*t * control_x + t**2 * x2
            y = (1-t)**2 * y1 + 2*(1-t)*t * control_y + t**2 * y2
            points.append((x, y))
        
        return points
    
    def calculate_duration(self, distance: float) -> float:
        """Calculate movement duration based on distance (Fitts's Law)."""
        # Fitts's Law: MT = a + b * log2(2D/W)
        # Simplified: longer distances take more time, but with diminishing returns
        base_time = 0.2  # Minimum time
        time_per_pixel = 0.001
        max_time = 2.0
        
        duration = base_time + (distance * time_per_pixel)
        
        # Add human variability
        duration *= random.uniform(0.8, 1.2)
        
        return min(duration, max_time)
    
    async def scroll(self, page, amount: int, direction: str = "down"):
        """Simulate human-like scrolling."""
        scroll_amount = -amount if direction == "up" else amount
        
        # Break into smaller scrolls with pauses
        chunk_size = random.randint(100, 300)
        num_chunks = abs(scroll_amount) // chunk_size
        
        for _ in range(num_chunks):
            await page.mouse.wheel(0, chunk_size if scroll_amount > 0 else -chunk_size)
            
            # Pause between scrolls
            await asyncio.sleep(random.uniform(0.1, 0.5))
            
            # Occasionally pause longer (reading)
            if random.random() < 0.1:
                await asyncio.sleep(random.uniform(0.5, 2.0))
```

### 10.2 Typing Simulation

```python
class TypingSimulator:
    """Simulates human-like typing patterns."""
    
    # Typing speed profiles (milliseconds per character)
    PROFILES = {
        "slow": {"mean": 200, "std": 50},
        "average": {"mean": 120, "std": 40},
        "fast": {"mean": 80, "std": 30},
        "professional": {"mean": 60, "std": 20}
    }
    
    # Common typing errors and corrections
    COMMON_ERRORS = {
        "a": ["s", "q", "z"],
        "e": ["r", "w", "d", "s"],
        "i": ["o", "u", "k", "j"],
        "o": ["i", "p", "l", "k"],
        "u": ["y", "i", "h", "j"],
        "n": ["b", "m", "h", "j"],
        "t": ["r", "y", "g", "f"],
    }
    
    def __init__(self, profile: str = "average"):
        self.profile = self.PROFILES[profile]
        self.error_rate = 0.02  # 2% error rate
    
    async def type_text(self, page, selector: str, text: str):
        """Type text with human-like timing and occasional errors."""
        for i, char in enumerate(text):
            # Decide whether to make a typo
            if random.random() < self.error_rate and char in self.COMMON_ERRORS:
                # Make a typo
                typo = random.choice(self.COMMON_ERRORS[char])
                await page.keyboard.press(typo)
                await self.get_delay()
                
                # Correct the typo
                await page.keyboard.press("Backspace")
                await self.get_delay()
            
            # Type the correct character
            await page.keyboard.press(char)
            
            # Get delay for next character
            delay = self.get_delay()
            
            # Add occasional longer pauses (thinking)
            if random.random() < 0.03:  # 3% chance
                delay += random.uniform(300, 800)
            
            # Add pause after punctuation
            if char in ".!":
                delay += random.uniform(200, 500)
            
            await asyncio.sleep(delay / 1000)
    
    def get_delay(self) -> float:
        """Get typing delay with human variability."""
        delay = random.gauss(self.profile["mean"], self.profile["std"])
        return max(delay, 20)  # Minimum 20ms
```

### 10.3 Behavior Patterns

```python
class BehaviorSimulator:
    """Orchestrates all human-like behavior simulation."""
    
    def __init__(self, config: BehaviorConfig):
        self.mouse = MouseSimulator()
        self.typing = TypingSimulator(config.typing_profile)
        self.reading_time = config.reading_time
        self.thinking_time = config.thinking_time
        self.scroll_behavior = config.scroll_behavior
    
    async def simulate_session(self, page, actions: List[UserAction]):
        """Simulate a complete user session."""
        for action in actions:
            # Pre-action delay (thinking)
            await self.think()
            
            # Execute action
            if action.type == "click":
                await self.simulate_click(page, action.target)
            elif action.type == "type":
                await self.simulate_typing(page, action.target, action.value)
            elif action.type == "scroll":
                await self.simulate_scroll(page, action.amount)
            elif action.type == "read":
                await self.simulate_reading(page, action.duration)
            
            # Post-action delay
            await self.pause()
    
    async def simulate_click(self, page, target: ElementHandle):
        """Simulate a human-like click."""
        # Get target position
        box = await target.bounding_box()
        target_x = box["x"] + box["width"] / 2
        target_y = box["y"] + box["height"] / 2
        
        # Add slight offset (humans don't click perfectly centered)
        target_x += random.gauss(0, box["width"] * 0.1)
        target_y += random.gauss(0, box["height"] * 0.1)
        
        # Move mouse to target
        await self.mouse.move_to(page, target_x, target_y)
        
        # Small pause before click
        await asyncio.sleep(random.uniform(0.05, 0.2))
        
        # Click
        await page.mouse.click(target_x, target_y)
        
        # Sometimes double-click
        if random.random() < 0.05:
            await asyncio.sleep(0.1)
            await page.mouse.click(target_x, target_y)
    
    async def simulate_typing(self, page, selector: str, text: str):
        """Simulate human-like typing."""
        # Click on field first
        await page.click(selector)
        await asyncio.sleep(0.1)
        
        # Type with human timing
        await self.typing.type_text(page, selector, text)
    
    async def simulate_reading(self, page, duration: Optional[float] = None):
        """Simulate reading time on a page."""
        if duration is None:
            # Calculate based on content
            content = await page.content()
            text_length = len(re.sub(r"<[^>]+>", "", content))
            
            # Average reading speed: 200-250 words per minute
            words = text_length / 5  # Approximate word count
            duration = (words / 225) * 60  # Convert to seconds
            
            # Add variability
            duration *= random.uniform(0.5, 1.5)
        
        # Occasionally scroll while reading
        end_time = time.time() + duration
        while time.time() < end_time:
            remaining = end_time - time.time()
            scroll_pause = min(random.uniform(3, 8), remaining)
            
            await asyncio.sleep(scroll_pause)
            
            if random.random() < 0.3 and remaining > 5:
                scroll_amount = random.randint(200, 500)
                await self.mouse.scroll(page, scroll_amount)
    
    async def think(self):
        """Simulate thinking time before action."""
        delay = random.gauss(self.thinking_time["mean"], 
                            self.thinking_time["std"])
        await asyncio.sleep(max(0, delay))
    
    async def pause(self):
        """Simulate pause after action."""
        delay = random.uniform(0.5, 2.0)
        await asyncio.sleep(delay)
```

---

## 11. SECURITY AND ANTI-DETECTION

### 11.1 Fingerprint Randomization

```python
class FingerprintManager:
    """Manages browser fingerprint randomization."""
    
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
    ]
    
    VIEWPORT_SIZES = [
        {"width": 1920, "height": 1080},
        {"width": 1366, "height": 768},
        {"width": 1536, "height": 864},
        {"width": 1440, "height": 900},
        {"width": 1280, "height": 720}
    ]
    
    def __init__(self):
        self.current_fingerprint = None
    
    def generate_fingerprint(self) -> BrowserFingerprint:
        """Generate a random but realistic browser fingerprint."""
        return BrowserFingerprint(
            user_agent=random.choice(self.USER_AGENTS),
            viewport=random.choice(self.VIEWPORT_SIZES),
            locale=random.choice(["en-US", "en-GB", "en-CA"]),
            timezone=random.choice(["America/New_York", "America/Chicago", 
                                   "America/Denver", "America/Los_Angeles"]),
            color_depth=24,
            device_memory=random.choice([4, 8, 16]),
            hardware_concurrency=random.choice([4, 8, 12, 16]),
            platform="Win32"
        )
    
    async def apply_fingerprint(self, context, fingerprint: BrowserFingerprint):
        """Apply fingerprint to browser context."""
        await context.set_extra_http_headers({
            "User-Agent": fingerprint.user_agent,
            "Accept-Language": fingerprint.locale
        })
        
        await context.add_init_script(f"""
            Object.defineProperty(navigator, "platform", {{
                get: () => "{fingerprint.platform}"
            }});
            Object.defineProperty(navigator, "hardwareConcurrency", {{
                get: () => {fingerprint.hardware_concurrency}
            }});
            Object.defineProperty(navigator, "deviceMemory", {{
                get: () => {fingerprint.device_memory}
            }});
        """)
```

### 11.2 Detection Evasion

```python
class EvasionTechniques:
    """Implements various anti-detection techniques."""
    
    STEALTH_SCRIPTS = {
        "webdriver": """
            Object.defineProperty(navigator, "webdriver", {
                get: () => undefined
            });
        """,
        "plugins": """
            Object.defineProperty(navigator, "plugins", {
                get: () => [
                    {
                        0: {type: "application/x-google-chrome-pdf", suffixes: "pdf", description: "Portable Document Format"},
                        description: "Portable Document Format",
                        filename: "internal-pdf-viewer",
                        length: 1,
                        name: "Chrome PDF Plugin"
                    },
                    {
                        0: {type: "application/pdf", suffixes: "pdf", description: ""},
                        description: "",
                        filename: "mhjfbmdgcfjbbpaeojofohoefgiehjai",
                        length: 1,
                        name: "Chrome PDF Viewer"
                    }
                ]
            });
        """,
        "chrome_runtime": """
            window.chrome = {
                runtime: {},
                loadTimes: function() {},
                csi: function() {},
                app: {}
            };
        """,
        "permissions": """
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === "notifications" ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
        """
    }
    
    async def apply_stealth(self, page):
        """Apply all stealth scripts to page."""
        for name, script in self.STEALTH_SCRIPTS.items():
            try:
                await page.add_init_script(script)
            except Exception as e:
                logger.warning(f"Failed to apply {name} stealth script: {e}")
    
    async def evade_bot_detection(self, page):
        """Apply comprehensive bot detection evasion."""
        # Apply stealth scripts
        await self.apply_stealth(page)
        
        # Override common detection methods
        await page.evaluate("""() => {
            // Override toString to hide modifications
            const originalToString = Function.prototype.toString;
            Function.prototype.toString = function() {
                if (this === Function.prototype.toString) {
                    return "function toString() { [native code] }";
                }
                return originalToString.call(this);
            };
            
            // Override console.debug to prevent detection
            const originalDebug = console.debug;
            console.debug = function() {};
            
            // Add fake notification permission
            if (window.Notification) {
                Object.defineProperty(Notification, "permission", {
                    get: () => "default"
                });
            }
        }""")
```

---

## 12. IMPLEMENTATION ROADMAP

### 12.1 Phase 1: Core Foundation (Weeks 1-2)
- [ ] DOM-based form detection
- [ ] Basic field classification (heuristics)
- [ ] Text input filling
- [ ] Simple mouse movement simulation

### 12.2 Phase 2: Enhanced Detection (Weeks 3-4)
- [ ] Visual form detection with OpenCV
- [ ] Hybrid detection system
- [ ] ML-based field classification
- [ ] Field relationship analysis

### 12.3 Phase 3: Advanced Interactions (Weeks 5-6)
- [ ] Dropdown and multi-select handling
- [ ] File upload automation
- [ ] Checkbox and radio button handling
- [ ] Date picker interactions

### 12.4 Phase 4: CAPTCHA Integration (Weeks 7-8)
- [ ] CAPTCHA detection system
- [ ] 2captcha integration
- [ ] Anti-captcha fallback
- [ ] reCAPTCHA v2/v3 handling

### 12.5 Phase 5: Behavior Simulation (Weeks 9-10)
- [ ] Bezier curve mouse movements
- [ ] Human-like typing patterns
- [ ] Reading time simulation
- [ ] Scroll behavior patterns

### 12.6 Phase 6: Anti-Detection (Weeks 11-12)
- [ ] Fingerprint randomization
- [ ] Stealth script injection
- [ ] Bot detection evasion
- [ ] Session management

---

## APPENDIX A: DATA MODELS

```python
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from enum import Enum

class FormType(Enum):
    LOGIN = "login"
    REGISTRATION = "registration"
    SEARCH = "search"
    PAYMENT = "payment"
    CONTACT = "contact"
    CHECKOUT = "checkout"
    SURVEY = "survey"
    GENERIC = "generic"

class CaptchaType(Enum):
    RECAPTCHA_V2 = "recaptcha_v2"
    RECAPTCHA_V3 = "recaptcha_v3"
    HCAPTCHA = "hcaptcha"
    IMAGE = "image_captcha"
    TEXT = "text_captcha"
    FUNCAPTCHA = "funcaptcha"

@dataclass
class FieldElement:
    id: str
    name: str
    type: str
    tag: str
    selector: str
    required: bool
    placeholder: Optional[str]
    bounds: Optional[Dict[str, float]]
    label_text: Optional[str] = None
    parent_element: Optional[str] = None
    fieldset: Optional[str] = None
    multiple: bool = False
    accept: Optional[str] = None
    max_file_size: Optional[int] = None
    readonly: bool = False
    disabled: bool = False

@dataclass
class FormElement:
    id: str
    selector: str
    action: Optional[str]
    method: str
    fields: List[FieldElement]
    bounds: Optional[Dict[str, float]]

@dataclass
class ClassificationResult:
    field_type: str
    confidence: float
    method: str

@dataclass
class ClassifiedField:
    element: FieldElement
    classification: ClassificationResult
    filled_value: Optional[Any] = None

@dataclass
class FillResult:
    success: bool
    field_id: str
    value: Optional[Any] = None
    error: Optional[str] = None
    strategy: Optional[str] = None

@dataclass
class FormFillResult:
    form_id: str
    field_results: List[FillResult]
    validation: Any
    success: bool

@dataclass
class CaptchaDetectionResult:
    detected: bool
    type: Optional[CaptchaType] = None
    confidence: float = 0.0
    element: Optional[Any] = None
    metadata: Dict = None

@dataclass
class CaptchaSolution:
    success: bool
    token: Optional[str] = None
    text: Optional[str] = None
    error: Optional[str] = None

@dataclass
class BrowserFingerprint:
    user_agent: str
    viewport: Dict[str, int]
    locale: str
    timezone: str
    color_depth: int
    device_memory: int
    hardware_concurrency: int
    platform: str
```

---

## APPENDIX B: CONFIGURATION SCHEMA

```yaml
# form_handling_config.yaml

form_detection:
  methods:
    - dom
    - visual
    - hybrid
  confidence_threshold: 0.7
  
field_classification:
  use_ml: true
  ml_model_path: "models/field_classifier.pkl"
  heuristic_weight: 0.4
  ml_weight: 0.4
  context_weight: 0.2
  
filling_strategies:
  default_profile: "average"
  profiles:
    slow:
      typing_speed: 200
      error_rate: 0.05
    average:
      typing_speed: 120
      error_rate: 0.02
    fast:
      typing_speed: 80
      error_rate: 0.01
      
behavior_simulation:
  enabled: true
  mouse_movement:
    bezier_curves: true
    jitter: 0.5
  typing:
    variable_delays: true
    occasional_errors: true
    thinking_pauses: true
  reading:
    simulate_reading_time: true
    scroll_while_reading: true
    
captcha:
  enabled: true
  providers:
    primary: "2captcha"
    fallback:
      - "anticaptcha"
      - "capsolver"
  api_keys:
    2captcha: "${TWOCAPTCHA_API_KEY}"
    anticaptcha: "${ANTICAPTCHA_API_KEY}"
  timeout: 120
  
anti_detection:
  enabled: true
  fingerprint_rotation: true
  stealth_scripts: true
  bot_evasion: true
  
logging:
  level: "INFO"
  capture_screenshots: true
  audit_trail: true
```

---

## DOCUMENT CONTROL

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-20 | AI Agent System | Initial specification |

---

*End of Technical Specification*
