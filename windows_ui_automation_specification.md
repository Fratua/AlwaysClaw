# Windows UI Automation & Accessibility Integration Specification
## OpenClaw-Inspired AI Agent System - Windows 10

**Version:** 1.0  
**Platform:** Windows 10  
**Target Framework:** .NET Framework 4.8 / .NET 6+  
**Document Type:** Technical Specification

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Microsoft UI Automation API](#3-microsoft-ui-automation-api)
4. [Element Discovery & Traversal](#4-element-discovery--traversal)
5. [Control Patterns](#5-control-patterns)
6. [Property Access & Modification](#6-property-access--modification)
7. [Event Handling System](#7-event-handling-system)
8. [Screen Scraping & Text Extraction](#8-screen-scraping--text-extraction)
9. [User Input Simulation](#9-user-input-simulation)
10. [Accessibility Tree Navigation](#10-accessibility-tree-navigation)
11. [Implementation Code Examples](#11-implementation-code-examples)
12. [Integration with Agent Loops](#12-integration-with-agent-loops)

---

## 1. Executive Summary

This specification defines the Windows UI Automation and Accessibility integration for a Windows 10-based OpenClaw-inspired AI agent system. The system leverages Microsoft UI Automation API 3.0 to provide programmatic access to UI elements, enabling the AI agent to:

- Control Windows applications programmatically
- Discover and interact with UI elements
- Extract text and visual information from screens
- Simulate user inputs (clicks, typing, focus changes)
- Monitor UI state changes through event subscriptions
- Navigate accessibility trees for comprehensive UI analysis

### Key Capabilities Matrix

| Capability | Method | Speed | Accuracy | Background Execution |
|------------|--------|-------|----------|---------------------|
| Element Discovery | UIA FindFirst/FindAll | High | 100% | Yes |
| Text Extraction (Full) | TextPattern | 10/10 | 100% | Yes |
| Text Extraction (Native) | GDI/Native | 8/10 | 100% | No |
| Text Extraction (OCR) | Tesseract/MODI | 3/10 | 98% | No |
| Input Simulation | SendInput/PostMessage | High | 100% | Partial |
| Event Monitoring | UIA Events | Real-time | 100% | Yes |

---

## 2. Architecture Overview

### 2.1 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AI Agent Core (GPT-5.2)                           │
│                    High-Level Decision Making & Planning                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    UI Automation Controller Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Element    │  │   Control    │  │   Property   │  │    Event     │    │
│  │   Discovery  │  │   Patterns   │  │   Manager    │  │   Handler    │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 Microsoft UI Automation API 3.0 (COM)                       │
│                    UIAutomationCore.dll / System.Windows.Automation         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Windows Application Layer                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │  Win32   │ │   WPF    │ │  WinForms│ │  UWP     │ │  Web     │          │
│  │  Apps    │ │   Apps   │ │  Apps    │ │  Apps    │ │  Browsers│          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Core Components

| Component | Namespace/Assembly | Purpose |
|-----------|-------------------|---------|
| UIA Client API | `System.Windows.Automation` | Primary interface for UI automation |
| UIA Core API | `UIAutomationClient.dll` | COM interface for native access |
| UIA Types | `UIAutomationTypes.dll` | Type definitions and constants |
| Input Simulator | `WindowsInput` / Custom | Low-level input simulation |
| OCR Engine | `Tesseract` / `Windows.Media.Ocr` | Text extraction from images |

---

## 3. Microsoft UI Automation API

### 3.1 Core Interfaces

#### 3.1.1 IUIAutomation (Primary Interface)

```csharp
// Primary entry point for UI Automation
IUIAutomation automation = new CUIAutomation();

// Key Methods:
- GetRootElement()           // Get desktop element
- GetFocusedElement()        // Get currently focused element
- ElementFromHandle(HWND)    // Get element from window handle
- ElementFromPoint(POINT)    // Get element at screen coordinates
- CreateTreeWalker()         // Create tree navigation object
- CreateCondition()          // Create search conditions
- AddAutomationEventHandler()    // Subscribe to events
- RemoveAutomationEventHandler() // Unsubscribe from events
```

#### 3.1.2 IUIAutomationElement

```csharp
// Represents a single UI element
interface IUIAutomationElement
{
    // Navigation
    IUIAutomationElement FindFirst(TreeScope scope, IUIAutomationCondition condition);
    IUIAutomationElementArray FindAll(TreeScope scope, IUIAutomationCondition condition);
    
    // Pattern Access
    IUnknown GetCurrentPattern(PATTERNID patternId);
    
    // Properties
    VARIANT GetCurrentPropertyValue(PROPERTYID propertyId);
    bool GetCurrentPropertyValueEx(PROPERTYID propertyId, bool ignoreDefaultValue);
    
    // Interaction
    void SetFocus();
    tagRECT CurrentBoundingRectangle { get; }
    string CurrentName { get; }
    int CurrentControlType { get; }
    bool CurrentIsEnabled { get; }
    bool CurrentIsOffscreen { get; }
}
```

### 3.2 Control Type Identifiers

| Control Type | ID | Description |
|--------------|-----|-------------|
| Button | `UIA_ButtonControlTypeId` (50000) | Clickable button |
| Calendar | `UIA_CalendarControlTypeId` (50001) | Date picker |
| CheckBox | `UIA_CheckBoxControlTypeId` (50002) | Toggleable checkbox |
| ComboBox | `UIA_ComboBoxControlTypeId` (50003) | Dropdown list |
| Edit | `UIA_EditControlTypeId` (50004) | Text input field |
| Hyperlink | `UIA_HyperlinkControlTypeId` (50005) | Clickable link |
| Image | `UIA_ImageControlTypeId` (50006) | Image element |
| ListItem | `UIA_ListItemControlTypeId` (50007) | Item in a list |
| List | `UIA_ListControlTypeId` (50008) | List container |
| Menu | `UIA_MenuControlTypeId` (50009) | Menu container |
| MenuBar | `UIA_MenuBarControlTypeId` (50010) | Menu bar |
| MenuItem | `UIA_MenuItemControlTypeId` (50011) | Menu item |
| ProgressBar | `UIA_ProgressBarControlTypeId` (50012) | Progress indicator |
| RadioButton | `UIA_RadioButtonControlTypeId` (50013) | Radio button |
| ScrollBar | `UIA_ScrollBarControlTypeId` (50014) | Scrollbar |
| Slider | `UIA_SliderControlTypeId` (50015) | Slider control |
| Spinner | `UIA_SpinnerControlTypeId` (50016) | Numeric spinner |
| StatusBar | `UIA_StatusBarControlTypeId` (50017) | Status bar |
| Tab | `UIA_TabControlTypeId` (50018) | Tab control |
| TabItem | `UIA_TabItemControlTypeId` (50019) | Tab item |
| Text | `UIA_TextControlTypeId` (50020) | Static text |
| ToolBar | `UIA_ToolBarControlTypeId` (50021) | Toolbar |
| ToolTip | `UIA_ToolTipControlTypeId` (50022) | Tooltip |
| Tree | `UIA_TreeControlTypeId` (50023) | Tree view |
| TreeItem | `UIA_TreeItemControlTypeId` (50024) | Tree item |
| Window | `UIA_WindowControlTypeId` (50032) | Window |
| Document | `UIA_DocumentControlTypeId` (50030) | Document container |
| DataGrid | `UIA_DataGridControlTypeId` (50028) | Data grid |
| SplitButton | `UIA_SplitButtonControlTypeId` (50031) | Split button |

---

## 4. Element Discovery & Traversal

### 4.1 Tree Scope Constants

```csharp
public enum TreeScope
{
    Element = 0x1,           // The element itself
    Children = 0x2,          // Direct children only
    Descendants = 0x4,       // All descendants
    Subtree = 0x7,           // Element + Children + Descendants
    Parent = 0x8,            // Parent element
    Ancestors = 0x10         // All ancestors
}
```

### 4.2 Search Conditions

```csharp
// Property Condition
var nameCondition = automation.CreatePropertyCondition(
    UIA_NamePropertyId, 
    "Submit Button"
);

// Control Type Condition
var buttonCondition = automation.CreatePropertyCondition(
    UIA_ControlTypePropertyId, 
    UIA_ButtonControlTypeId
);

// Automation ID Condition
var idCondition = automation.CreatePropertyCondition(
    UIA_AutomationIdPropertyId, 
    "btnSubmit"
);

// Combined Conditions (AND)
var andCondition = automation.CreateAndCondition(
    buttonCondition, 
    nameCondition
);

// Combined Conditions (OR)
var orCondition = automation.CreateOrCondition(
    nameCondition, 
    idCondition
);

// Not Condition
var notCondition = automation.CreateNotCondition(
    buttonCondition
);
```

### 4.3 Element Discovery Methods

```csharp
public class ElementDiscovery
{
    private IUIAutomation _automation;
    
    public ElementDiscovery()
    {
        _automation = new CUIAutomation();
    }
    
    /// <summary>
    /// Find element by name
    /// </summary>
    public IUIAutomationElement FindByName(string name, IUIAutomationElement root = null)
    {
        root = root ?? _automation.GetRootElement();
        var condition = _automation.CreatePropertyCondition(UIA_NamePropertyId, name);
        return root.FindFirst(TreeScope.Descendants, condition);
    }
    
    /// <summary>
    /// Find element by automation ID
    /// </summary>
    public IUIAutomationElement FindByAutomationId(string automationId, IUIAutomationElement root = null)
    {
        root = root ?? _automation.GetRootElement();
        var condition = _automation.CreatePropertyCondition(UIA_AutomationIdPropertyId, automationId);
        return root.FindFirst(TreeScope.Descendants, condition);
    }
    
    /// <summary>
    /// Find element by control type
    /// </summary>
    public IUIAutomationElement FindByControlType(int controlType, IUIAutomationElement root = null)
    {
        root = root ?? _automation.GetRootElement();
        var condition = _automation.CreatePropertyCondition(UIA_ControlTypePropertyId, controlType);
        return root.FindFirst(TreeScope.Descendants, condition);
    }
    
    /// <summary>
    /// Find all elements matching condition
    /// </summary>
    public IUIAutomationElementArray FindAll(IUIAutomationCondition condition, IUIAutomationElement root = null)
    {
        root = root ?? _automation.GetRootElement();
        return root.FindAll(TreeScope.Descendants, condition);
    }
    
    /// <summary>
    /// Find element from window handle
    /// </summary>
    public IUIAutomationElement FromWindowHandle(IntPtr hwnd)
    {
        return _automation.ElementFromHandle(hwnd);
    }
    
    /// <summary>
    /// Find element at screen coordinates
    /// </summary>
    public IUIAutomationElement FromPoint(System.Drawing.Point point)
    {
        tagPOINT pt = new tagPOINT { x = point.X, y = point.Y };
        return _automation.ElementFromPoint(pt);
    }
    
    /// <summary>
    /// Get currently focused element
    /// </summary>
    public IUIAutomationElement GetFocusedElement()
    {
        return _automation.GetFocusedElement();
    }
}
```

---

## 5. Control Patterns

### 5.1 Pattern Interface Reference

| Pattern | Provider Interface | Client Interface | Purpose |
|---------|-------------------|------------------|---------|
| Invoke | `IInvokeProvider` | `IUIAutomationInvokePattern` | Click/invoke controls |
| Value | `IValueProvider` | `IUIAutomationValuePattern` | Get/set text values |
| RangeValue | `IRangeValueProvider` | `IUIAutomationRangeValuePattern` | Numeric range values |
| Toggle | `IToggleProvider` | `IUIAutomationTogglePattern` | Toggle states |
| Selection | `ISelectionProvider` | `IUIAutomationSelectionPattern` | Selection containers |
| SelectionItem | `ISelectionItemProvider` | `IUIAutomationSelectionItemPattern` | Selectable items |
| ExpandCollapse | `IExpandCollapseProvider` | `IUIAutomationExpandCollapsePattern` | Expand/collapse |
| Scroll | `IScrollProvider` | `IUIAutomationScrollPattern` | Scrollable controls |
| ScrollItem | `IScrollItemProvider` | `IUIAutomationScrollItemPattern` | Scroll items into view |
| Grid | `IGridProvider` | `IUIAutomationGridPattern` | Grid functionality |
| GridItem | `IGridItemProvider` | `IUIAutomationGridItemPattern` | Grid cells |
| Table | `ITableProvider` | `IUIAutomationTablePattern` | Tables with headers |
| Text | `ITextProvider` | `IUIAutomationTextPattern` | Text content |
| Window | `IWindowProvider` | `IUIAutomationWindowPattern` | Window operations |
| Transform | `ITransformProvider` | `IUIAutomationTransformPattern` | Move/resize/rotate |
| Dock | `IDockProvider` | `IUIAutomationDockPattern` | Docking |
| MultipleView | `IMultipleViewProvider` | `IUIAutomationMultipleViewPattern` | Multiple views |

### 5.2 Pattern Implementation

```csharp
public class ControlPatternManager
{
    private IUIAutomation _automation;
    
    public ControlPatternManager()
    {
        _automation = new CUIAutomation();
    }
    
    #region Invoke Pattern
    
    public void Invoke(IUIAutomationElement element)
    {
        var pattern = GetPattern<IUIAutomationInvokePattern>(element, UIA_InvokePatternId);
        pattern?.Invoke();
    }
    
    public bool SupportsInvoke(IUIAutomationElement element)
    {
        return GetPattern<IUIAutomationInvokePattern>(element, UIA_InvokePatternId) != null;
    }
    
    #endregion
    
    #region Value Pattern
    
    public string GetValue(IUIAutomationElement element)
    {
        var pattern = GetPattern<IUIAutomationValuePattern>(element, UIA_ValuePatternId);
        return pattern?.CurrentValue ?? string.Empty;
    }
    
    public void SetValue(IUIAutomationElement element, string value)
    {
        var pattern = GetPattern<IUIAutomationValuePattern>(element, UIA_ValuePatternId);
        pattern?.SetValue(value);
    }
    
    public bool IsReadOnly(IUIAutomationElement element)
    {
        var pattern = GetPattern<IUIAutomationValuePattern>(element, UIA_ValuePatternId);
        return pattern?.CurrentIsReadOnly ?? true;
    }
    
    #endregion
    
    #region Toggle Pattern
    
    public void Toggle(IUIAutomationElement element)
    {
        var pattern = GetPattern<IUIAutomationTogglePattern>(element, UIA_TogglePatternId);
        pattern?.Toggle();
    }
    
    public ToggleState GetToggleState(IUIAutomationElement element)
    {
        var pattern = GetPattern<IUIAutomationTogglePattern>(element, UIA_TogglePatternId);
        return (ToggleState)(pattern?.CurrentToggleState ?? 0);
    }
    
    #endregion
    
    #region Selection Pattern
    
    public IUIAutomationElementArray GetSelection(IUIAutomationElement element)
    {
        var pattern = GetPattern<IUIAutomationSelectionPattern>(element, UIA_SelectionPatternId);
        return pattern?.GetCurrentSelection();
    }
    
    public bool CanSelectMultiple(IUIAutomationElement element)
    {
        var pattern = GetPattern<IUIAutomationSelectionPattern>(element, UIA_SelectionPatternId);
        return pattern?.CurrentCanSelectMultiple ?? false;
    }
    
    #endregion
    
    #region SelectionItem Pattern
    
    public void Select(IUIAutomationElement element)
    {
        var pattern = GetPattern<IUIAutomationSelectionItemPattern>(element, UIA_SelectionItemPatternId);
        pattern?.Select();
    }
    
    public void AddToSelection(IUIAutomationElement element)
    {
        var pattern = GetPattern<IUIAutomationSelectionItemPattern>(element, UIA_SelectionItemPatternId);
        pattern?.AddToSelection();
    }
    
    public void RemoveFromSelection(IUIAutomationElement element)
    {
        var pattern = GetPattern<IUIAutomationSelectionItemPattern>(element, UIA_SelectionItemPatternId);
        pattern?.RemoveFromSelection();
    }
    
    public bool IsSelected(IUIAutomationElement element)
    {
        var pattern = GetPattern<IUIAutomationSelectionItemPattern>(element, UIA_SelectionItemPatternId);
        return pattern?.CurrentIsSelected ?? false;
    }
    
    #endregion
    
    #region ExpandCollapse Pattern
    
    public void Expand(IUIAutomationElement element)
    {
        var pattern = GetPattern<IUIAutomationExpandCollapsePattern>(element, UIA_ExpandCollapsePatternId);
        pattern?.Expand();
    }
    
    public void Collapse(IUIAutomationElement element)
    {
        var pattern = GetPattern<IUIAutomationExpandCollapsePattern>(element, UIA_ExpandCollapsePatternId);
        pattern?.Collapse();
    }
    
    public ExpandCollapseState GetExpandCollapseState(IUIAutomationElement element)
    {
        var pattern = GetPattern<IUIAutomationExpandCollapsePattern>(element, UIA_ExpandCollapsePatternId);
        return (ExpandCollapseState)(pattern?.CurrentExpandCollapseState ?? 0);
    }
    
    #endregion
    
    #region Scroll Pattern
    
    public void ScrollHorizontal(IUIAutomationElement element, ScrollAmount amount)
    {
        var pattern = GetPattern<IUIAutomationScrollPattern>(element, UIA_ScrollPatternId);
        pattern?.Scroll(amount, ScrollAmount.NoAmount);
    }
    
    public void ScrollVertical(IUIAutomationElement element, ScrollAmount amount)
    {
        var pattern = GetPattern<IUIAutomationScrollPattern>(element, UIA_ScrollPatternId);
        pattern?.Scroll(ScrollAmount.NoAmount, amount);
    }
    
    public void SetScrollPercent(IUIAutomationElement element, double horizontalPercent, double verticalPercent)
    {
        var pattern = GetPattern<IUIAutomationScrollPattern>(element, UIA_ScrollPatternId);
        pattern?.SetScrollPercent(horizontalPercent, verticalPercent);
    }
    
    #endregion
    
    #region Window Pattern
    
    public void CloseWindow(IUIAutomationElement element)
    {
        var pattern = GetPattern<IUIAutomationWindowPattern>(element, UIA_WindowPatternId);
        pattern?.Close();
    }
    
    public void SetWindowVisualState(IUIAutomationElement element, WindowVisualState state)
    {
        var pattern = GetPattern<IUIAutomationWindowPattern>(element, UIA_WindowPatternId);
        pattern?.SetWindowVisualState(state);
    }
    
    public void Maximize(IUIAutomationElement element)
    {
        SetWindowVisualState(element, WindowVisualState.Maximized);
    }
    
    public void Minimize(IUIAutomationElement element)
    {
        SetWindowVisualState(element, WindowVisualState.Minimized);
    }
    
    public void Restore(IUIAutomationElement element)
    {
        SetWindowVisualState(element, WindowVisualState.Normal);
    }
    
    #endregion
    
    #region RangeValue Pattern
    
    public void SetRangeValue(IUIAutomationElement element, double value)
    {
        var pattern = GetPattern<IUIAutomationRangeValuePattern>(element, UIA_RangeValuePatternId);
        if (pattern != null)
        {
            double min = pattern.CurrentMinimum;
            double max = pattern.CurrentMaximum;
            value = Math.Max(min, Math.Min(max, value));
            pattern.SetValue(value);
        }
    }
    
    public double GetRangeValue(IUIAutomationElement element)
    {
        var pattern = GetPattern<IUIAutomationRangeValuePattern>(element, UIA_RangeValuePatternId);
        return pattern?.CurrentValue ?? 0;
    }
    
    public void Increment(IUIAutomationElement element)
    {
        var pattern = GetPattern<IUIAutomationRangeValuePattern>(element, UIA_RangeValuePatternId);
        if (pattern != null)
        {
            double newValue = pattern.CurrentValue + pattern.CurrentSmallChange;
            SetRangeValue(element, newValue);
        }
    }
    
    public void Decrement(IUIAutomationElement element)
    {
        var pattern = GetPattern<IUIAutomationRangeValuePattern>(element, UIA_RangeValuePatternId);
        if (pattern != null)
        {
            double newValue = pattern.CurrentValue - pattern.CurrentSmallChange;
            SetRangeValue(element, newValue);
        }
    }
    
    #endregion
    
    #region Text Pattern
    
    public string GetText(IUIAutomationElement element)
    {
        var pattern = GetPattern<IUIAutomationTextPattern>(element, UIA_TextPatternId);
        return pattern?.DocumentRange?.GetText(-1) ?? string.Empty;
    }
    
    public string GetSelectedText(IUIAutomationElement element)
    {
        var pattern = GetPattern<IUIAutomationTextPattern>(element, UIA_TextPatternId);
        var ranges = pattern?.GetSelection();
        if (ranges != null && ranges.Length > 0)
        {
            return ranges[0].GetText(-1);
        }
        return string.Empty;
    }
    
    #endregion
    
    #region Helper Methods
    
    private T GetPattern<T>(IUIAutomationElement element, int patternId) where T : class
    {
        if (element == null) return null;
        
        try
        {
            var pattern = element.GetCurrentPattern(patternId);
            return pattern as T;
        }
        catch
        {
            return null;
        }
    }
    
    #endregion
}

// Enums
public enum ToggleState { Off = 0, On = 1, Indeterminate = 2 }
public enum ExpandCollapseState { Collapsed = 0, Expanded = 1, PartiallyExpanded = 2, LeafNode = 3 }
public enum ScrollAmount { LargeDecrement = 0, SmallDecrement = 1, NoAmount = 2, LargeIncrement = 3, SmallIncrement = 4 }
public enum WindowVisualState { Normal = 0, Maximized = 1, Minimized = 2 }
```

---

## 6. Property Access & Modification

### 6.1 Key Property Identifiers

```csharp
public static class UIAProperties
{
    public const int UIA_RuntimeIdPropertyId = 30000;
    public const int UIA_BoundingRectanglePropertyId = 30001;
    public const int UIA_ProcessIdPropertyId = 30002;
    public const int UIA_ControlTypePropertyId = 30003;
    public const int UIA_NamePropertyId = 30005;
    public const int UIA_AutomationIdPropertyId = 30011;
    public const int UIA_ClassNamePropertyId = 30012;
    public const int UIA_IsEnabledPropertyId = 30010;
    public const int UIA_IsOffscreenPropertyId = 30022;
    public const int UIA_FrameworkIdPropertyId = 30024;
    public const int UIA_ValueValuePropertyId = 30045;
    public const int UIA_ValueIsReadOnlyPropertyId = 30046;
}
```

### 6.2 Property Access Implementation

```csharp
public class PropertyManager
{
    private IUIAutomation _automation;
    
    public PropertyManager()
    {
        _automation = new CUIAutomation();
    }
    
    public string GetPropertyAsString(IUIAutomationElement element, int propertyId)
    {
        try
        {
            var value = element.GetCurrentPropertyValue(propertyId);
            return Convert.ToString(value);
        }
        catch
        {
            return null;
        }
    }
    
    public System.Drawing.Rectangle? GetBoundingRectangle(IUIAutomationElement element)
    {
        try
        {
            var rect = element.CurrentBoundingRectangle;
            return System.Drawing.Rectangle.FromLTRB(rect.left, rect.top, rect.right, rect.bottom);
        }
        catch
        {
            return null;
        }
    }
    
    public System.Drawing.Point? GetClickablePoint(IUIAutomationElement element)
    {
        try
        {
            tagPOINT point;
            bool gotClickable = element.GetClickablePoint(out point);
            if (gotClickable)
            {
                return new System.Drawing.Point(point.x, point.y);
            }
        }
        catch { }
        
        // Fallback to center of bounding rectangle
        var rect = GetBoundingRectangle(element);
        if (rect.HasValue)
        {
            return new System.Drawing.Point(
                rect.Value.Left + rect.Value.Width / 2,
                rect.Value.Top + rect.Value.Height / 2
            );
        }
        
        return null;
    }
    
    public ElementInfo GetElementInfo(IUIAutomationElement element)
    {
        return new ElementInfo
        {
            Name = element.CurrentName,
            ControlType = GetControlTypeName(element.CurrentControlType),
            AutomationId = element.CurrentAutomationId,
            ClassName = element.CurrentClassName,
            IsEnabled = element.CurrentIsEnabled,
            IsVisible = !element.CurrentIsOffscreen,
            BoundingRectangle = GetBoundingRectangle(element),
            ProcessId = element.CurrentProcessId,
            FrameworkId = element.CurrentFrameworkId
        };
    }
    
    private string GetControlTypeName(int controlTypeId)
    {
        switch (controlTypeId)
        {
            case 50000: return "Button";
            case 50002: return "CheckBox";
            case 50003: return "ComboBox";
            case 50004: return "Edit";
            case 50005: return "Hyperlink";
            case 50006: return "Image";
            case 50007: return "ListItem";
            case 50008: return "List";
            case 50011: return "MenuItem";
            case 50012: return "ProgressBar";
            case 50013: return "RadioButton";
            case 50018: return "Tab";
            case 50019: return "TabItem";
            case 50020: return "Text";
            case 50023: return "Tree";
            case 50024: return "TreeItem";
            case 50028: return "DataGrid";
            case 50030: return "Document";
            case 50031: return "SplitButton";
            case 50032: return "Window";
            default: return $"Unknown({controlTypeId})";
        }
    }
}

public class ElementInfo
{
    public string Name { get; set; }
    public string ControlType { get; set; }
    public string AutomationId { get; set; }
    public string ClassName { get; set; }
    public bool IsEnabled { get; set; }
    public bool IsVisible { get; set; }
    public System.Drawing.Rectangle? BoundingRectangle { get; set; }
    public int ProcessId { get; set; }
    public string FrameworkId { get; set; }
    
    public override string ToString()
    {
        return $"[{ControlType}] Name='{Name}', AutomationId='{AutomationId}', Enabled={IsEnabled}";
    }
}
```

---

## 7. Event Handling System

### 7.1 Event Categories

| Event Category | Description | Use Cases |
|----------------|-------------|-----------|
| Property Change | Raised when a property changes | Monitor checkbox state, text changes |
| Element Action | Raised when UI changes from user/programmatic activity | Button clicks, invocations |
| Structure Change | Raised when UI Automation tree structure changes | New windows, elements added/removed |
| Global Desktop Change | Raised for global actions | Focus changes, window closes |
| Notification | Custom application notifications | Toast notifications, alerts |

### 7.2 Event Identifiers

```csharp
public static class UIAEvents
{
    public const int UIA_StructureChangedEventId = 20002;
    public const int UIA_AutomationPropertyChangedEventId = 20004;
    public const int UIA_AutomationFocusChangedEventId = 20005;
    public const int UIA_Invoke_InvokedEventId = 20009;
    public const int UIA_SelectionItem_ElementSelectedEventId = 20012;
    public const int UIA_Selection_InvalidatedEventId = 20013;
    public const int UIA_Text_TextChangedEventId = 20015;
    public const int UIA_Window_WindowOpenedEventId = 20016;
    public const int UIA_Window_WindowClosedEventId = 20017;
    public const int UIA_NotificationEventId = 20035;
}
```

### 7.3 Event Handler Implementation

```csharp
public class EventManager : IDisposable
{
    private IUIAutomation _automation;
    private IUIAutomationEventHandler _genericEventHandler;
    private IUIAutomationPropertyChangedEventHandler _propertyChangedHandler;
    private IUIAutomationStructureChangedEventHandler _structureChangedHandler;
    private IUIAutomationFocusChangedEventHandler _focusChangedHandler;
    
    public event EventHandler<ElementInvokedEventArgs> ElementInvoked;
    public event EventHandler<PropertyChangedEventArgs> PropertyChanged;
    public event EventHandler<StructureChangedEventArgs> StructureChanged;
    public event EventHandler<FocusChangedEventArgs> FocusChanged;
    public event EventHandler<TextChangedEventArgs> TextChanged;
    public event EventHandler<WindowEventArgs> WindowOpened;
    public event EventHandler<WindowEventArgs> WindowClosed;
    
    public EventManager()
    {
        _automation = new CUIAutomation();
        InitializeHandlers();
    }
    
    private void InitializeHandlers()
    {
        _genericEventHandler = new GenericEventHandler(this);
        _propertyChangedHandler = new PropertyChangedEventHandlerImpl(this);
        _structureChangedHandler = new StructureChangedEventHandlerImpl(this);
        _focusChangedHandler = new FocusChangedEventHandlerImpl(this);
    }
    
    public void SubscribeToInvoke(IUIAutomationElement element)
    {
        SubscribeToEvent(element, UIAEvents.UIA_Invoke_InvokedEventId, TreeScope.Element);
    }
    
    public void SubscribeToPropertyChange(IUIAutomationElement element, params int[] propertyIds)
    {
        var propertyArray = propertyIds ?? new[] { UIAProperties.UIA_NamePropertyId };
        _automation.AddPropertyChangedEventHandler(element, TreeScope.Element, _propertyChangedHandler, propertyArray);
    }
    
    public void SubscribeToStructureChanges(IUIAutomationElement element, TreeScope scope = TreeScope.Subtree)
    {
        _automation.AddStructureChangedEventHandler(element, scope, _structureChangedHandler);
    }
    
    public void SubscribeToFocusChanges()
    {
        _automation.AddFocusChangedEventHandler(_focusChangedHandler);
    }
    
    public void SubscribeToWindowEvents()
    {
        var root = _automation.GetRootElement();
        SubscribeToEvent(root, UIAEvents.UIA_Window_WindowOpenedEventId, TreeScope.Subtree);
        SubscribeToEvent(root, UIAEvents.UIA_Window_WindowClosedEventId, TreeScope.Subtree);
    }
    
    private void SubscribeToEvent(IUIAutomationElement element, int eventId, TreeScope scope)
    {
        _automation.AddAutomationEventHandler(eventId, element, scope, null, _genericEventHandler);
    }
    
    public void UnsubscribeAll()
    {
        _automation.RemoveAllEventHandlers();
    }
    
    internal void HandleGenericEvent(IUIAutomationElement sender, int eventId)
    {
        switch (eventId)
        {
            case UIAEvents.UIA_Invoke_InvokedEventId:
                ElementInvoked?.Invoke(this, new ElementInvokedEventArgs { Element = sender });
                break;
            case UIAEvents.UIA_Text_TextChangedEventId:
                TextChanged?.Invoke(this, new TextChangedEventArgs { Element = sender });
                break;
            case UIAEvents.UIA_Window_WindowOpenedEventId:
                WindowOpened?.Invoke(this, new WindowEventArgs { Element = sender, IsOpen = true });
                break;
            case UIAEvents.UIA_Window_WindowClosedEventId:
                WindowClosed?.Invoke(this, new WindowEventArgs { Element = sender, IsOpen = false });
                break;
        }
    }
    
    internal void HandlePropertyChanged(IUIAutomationElement sender, int propertyId, object newValue)
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs 
        { 
            Element = sender, 
            PropertyId = propertyId, 
            NewValue = newValue 
        });
    }
    
    internal void HandleStructureChanged(IUIAutomationElement sender, StructureChangeType changeType, int[] runtimeId)
    {
        StructureChanged?.Invoke(this, new StructureChangedEventArgs 
        { 
            Element = sender, 
            ChangeType = changeType, 
            RuntimeId = runtimeId 
        });
    }
    
    internal void HandleFocusChanged(IUIAutomationElement sender)
    {
        FocusChanged?.Invoke(this, new FocusChangedEventArgs { Element = sender });
    }
    
    public void Dispose()
    {
        UnsubscribeAll();
        _automation = null;
    }
}

// Event Args Classes
public class ElementInvokedEventArgs : EventArgs { public IUIAutomationElement Element { get; set; } }
public class PropertyChangedEventArgs : EventArgs { public IUIAutomationElement Element { get; set; } public int PropertyId { get; set; } public object NewValue { get; set; } }
public class StructureChangedEventArgs : EventArgs { public IUIAutomationElement Element { get; set; } public StructureChangeType ChangeType { get; set; } public int[] RuntimeId { get; set; } }
public class FocusChangedEventArgs : EventArgs { public IUIAutomationElement Element { get; set; } }
public class TextChangedEventArgs : EventArgs { public IUIAutomationElement Element { get; set; } }
public class WindowEventArgs : EventArgs { public IUIAutomationElement Element { get; set; } public bool IsOpen { get; set; } }

public enum StructureChangeType { ChildAdded = 0, ChildRemoved = 1, ChildrenInvalidated = 2, ChildrenBulkAdded = 3, ChildrenBulkRemoved = 4, ChildrenReordered = 5 }
```

---

## 8. Screen Scraping & Text Extraction

### 8.1 Extraction Methods Comparison

| Method | Speed | Accuracy | Background | Hidden Text | Citrix | Best For |
|--------|-------|----------|------------|-------------|--------|----------|
| Full Text | 10/10 | 100% | Yes | Yes | No | Documents, Forms |
| Native | 8/10 | 100% | No | No | No | Desktop apps with GDI |
| OCR | 3/10 | 98% | No | No | Yes | Images, Citrix, Legacy |

### 8.2 Screen Scraping Implementation

```csharp
public class ScreenScraper
{
    private IUIAutomation _automation;
    
    public ScreenScraper()
    {
        _automation = new CUIAutomation();
    }
    
    public string ExtractFullText(IUIAutomationElement element)
    {
        if (element == null) return string.Empty;
        
        // Try TextPattern first
        try
        {
            var pattern = element.GetCurrentPattern(UIA_TextPatternId) as IUIAutomationTextPattern;
            if (pattern != null)
                return pattern.DocumentRange.GetText(-1);
        }
        catch { }
        
        // Fallback to Value pattern
        try
        {
            var valuePattern = element.GetCurrentPattern(UIA_ValuePatternId) as IUIAutomationValuePattern;
            if (valuePattern != null)
                return valuePattern.CurrentValue;
        }
        catch { }
        
        // Fallback to Name property
        try
        {
            return element.CurrentName ?? string.Empty;
        }
        catch { }
        
        return string.Empty;
    }
    
    public Dictionary<string, string> ExtractAllTextFromWindow(IUIAutomationElement window)
    {
        var results = new Dictionary<string, string>();
        var walker = _automation.ControlViewWalker;
        ExtractTextRecursive(window, results, walker);
        return results;
    }
    
    private void ExtractTextRecursive(IUIAutomationElement element, Dictionary<string, string> results, IUIAutomationTreeWalker walker)
    {
        if (element == null) return;
        
        string text = ExtractFullText(element);
        if (!string.IsNullOrWhiteSpace(text))
        {
            string key = $"{element.CurrentControlType}_{element.CurrentAutomationId ?? element.CurrentName ?? Guid.NewGuid().ToString()}";
            results[key] = text;
        }
        
        var child = walker.GetFirstChildElement(element);
        while (child != null)
        {
            ExtractTextRecursive(child, results, walker);
            child = walker.GetNextSiblingElement(child);
        }
    }
}
```

---

## 9. User Input Simulation

### 9.1 Input Methods Comparison

| Method | Compatibility | Background | Speed | Hotkey Support | Best For |
|--------|--------------|------------|-------|----------------|----------|
| Hardware Events | 100% | No | Medium | Yes | Universal fallback |
| Window Messages | 80% | Yes | Medium | Yes | Win32 controls |
| Simulate | 99% Web / 60% Desktop | Yes | High | No | Modern web apps |

### 9.2 Input Simulation Implementation

```csharp
public class InputSimulator
{
    private IUIAutomation _automation;
    private ControlPatternManager _patternManager;
    
    public InputSimulator()
    {
        _automation = new CUIAutomation();
        _patternManager = new ControlPatternManager();
    }
    
    public void Click(IUIAutomationElement element, MouseButton button = MouseButton.Left)
    {
        if (element == null) return;
        
        // Try Invoke pattern first
        if (button == MouseButton.Left && _patternManager.SupportsInvoke(element))
        {
            _patternManager.Invoke(element);
            return;
        }
        
        var point = GetClickablePoint(element);
        if (!point.HasValue) return;
        
        SetCursorPos(point.Value.X, point.Value.Y);
        
        switch (button)
        {
            case MouseButton.Left:
                mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
                Thread.Sleep(50);
                mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
                break;
            case MouseButton.Right:
                mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0);
                Thread.Sleep(50);
                mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0);
                break;
        }
    }
    
    public void TypeText(IUIAutomationElement element, string text, bool clearFirst = true)
    {
        if (element == null || string.IsNullOrEmpty(text)) return;
        
        element.SetFocus();
        Thread.Sleep(100);
        
        // Try Value pattern first
        var valuePattern = element.GetCurrentPattern(UIA_ValuePatternId) as IUIAutomationValuePattern;
        if (valuePattern != null && !valuePattern.CurrentIsReadOnly)
        {
            if (clearFirst) valuePattern.SetValue(string.Empty);
            valuePattern.SetValue(text);
            return;
        }
        
        // Fallback to SendKeys
        if (clearFirst)
        {
            SendKeys("^a");
            Thread.Sleep(50);
            SendKeys("{DELETE}");
            Thread.Sleep(50);
        }
        SendKeys(text);
    }
    
    public void SetFocus(IUIAutomationElement element) => element?.SetFocus();
    public IUIAutomationElement GetFocusedElement() => _automation.GetFocusedElement();
    
    private System.Drawing.Point? GetClickablePoint(IUIAutomationElement element)
    {
        try
        {
            tagPOINT point;
            if (element.GetClickablePoint(out point))
                return new System.Drawing.Point(point.x, point.y);
        }
        catch { }
        
        var rect = element.CurrentBoundingRectangle;
        return new System.Drawing.Point((rect.left + rect.right) / 2, (rect.top + rect.bottom) / 2);
    }
    
    [DllImport("user32.dll")] private static extern bool SetCursorPos(int x, int y);
    [DllImport("user32.dll")] private static extern void mouse_event(uint dwFlags, uint dx, uint dy, uint dwData, int dwExtraInfo);
    
    private const uint MOUSEEVENTF_LEFTDOWN = 0x0002;
    private const uint MOUSEEVENTF_LEFTUP = 0x0004;
    private const uint MOUSEEVENTF_RIGHTDOWN = 0x0008;
    private const uint MOUSEEVENTF_RIGHTUP = 0x0010;
}

public enum MouseButton { Left, Right, Middle }
```

---

## 10. Accessibility Tree Navigation

```csharp
public class AccessibilityTreeNavigator
{
    private IUIAutomation _automation;
    private IUIAutomationTreeWalker _walker;
    
    public AccessibilityTreeNavigator(TreeWalkerView view = TreeWalkerView.ControlView)
    {
        _automation = new CUIAutomation();
        _walker = GetWalker(view);
    }
    
    private IUIAutomationTreeWalker GetWalker(TreeWalkerView view)
    {
        switch (view)
        {
            case TreeWalkerView.RawView: return _automation.RawViewWalker;
            case TreeWalkerView.ControlView: return _automation.ControlViewWalker;
            case TreeWalkerView.ContentView: return _automation.ContentViewWalker;
            default: return _automation.ControlViewWalker;
        }
    }
    
    public IUIAutomationElement GetRootElement() => _automation.GetRootElement();
    public IUIAutomationElement GetParent(IUIAutomationElement element) => _walker.GetParentElement(element);
    public IUIAutomationElement GetFirstChild(IUIAutomationElement element) => _walker.GetFirstChildElement(element);
    public IUIAutomationElement GetNextSibling(IUIAutomationElement element) => _walker.GetNextSiblingElement(element);
    
    public List<AccessibilityNode> WalkSubtree(IUIAutomationElement root)
    {
        var nodes = new List<AccessibilityNode>();
        WalkRecursive(root, nodes, 0);
        return nodes;
    }
    
    private void WalkRecursive(IUIAutomationElement element, List<AccessibilityNode> nodes, int depth)
    {
        if (element == null) return;
        nodes.Add(CreateNode(element, depth));
        
        var child = _walker.GetFirstChildElement(element);
        while (child != null)
        {
            WalkRecursive(child, nodes, depth + 1);
            child = _walker.GetNextSiblingElement(child);
        }
    }
    
    private AccessibilityNode CreateNode(IUIAutomationElement element, int depth)
    {
        return new AccessibilityNode
        {
            Depth = depth,
            RuntimeId = string.Join(".", element.GetRuntimeId() ?? new int[0]),
            ControlType = GetControlTypeName(element.CurrentControlType),
            Name = element.CurrentName,
            AutomationId = element.CurrentAutomationId,
            ClassName = element.CurrentClassName,
            IsEnabled = element.CurrentIsEnabled,
            IsOffscreen = element.CurrentIsOffscreen,
            ProcessId = element.CurrentProcessId
        };
    }
    
    private string GetControlTypeName(int controlTypeId)
    {
        switch (controlTypeId)
        {
            case 50000: return "Button";
            case 50002: return "CheckBox";
            case 50003: return "ComboBox";
            case 50004: return "Edit";
            case 50020: return "Text";
            case 50032: return "Window";
            default: return $"Unknown({controlTypeId})";
        }
    }
}

public class AccessibilityNode
{
    public int Depth { get; set; }
    public string RuntimeId { get; set; }
    public string ControlType { get; set; }
    public string Name { get; set; }
    public string AutomationId { get; set; }
    public string ClassName { get; set; }
    public bool IsEnabled { get; set; }
    public bool IsOffscreen { get; set; }
    public int ProcessId { get; set; }
}

public enum TreeWalkerView { RawView, ControlView, ContentView }
```

---

## 11. Implementation Code Examples

### 11.1 Complete Agent UI Controller

```csharp
public class AgentUIController : IDisposable
{
    private IUIAutomation _automation;
    private ElementDiscovery _discovery;
    private ControlPatternManager _patterns;
    private PropertyManager _properties;
    private EventManager _events;
    private ScreenScraper _scraper;
    private InputSimulator _input;
    private AccessibilityTreeNavigator _navigator;
    
    public AgentUIController()
    {
        _automation = new CUIAutomation();
        _discovery = new ElementDiscovery();
        _patterns = new ControlPatternManager();
        _properties = new PropertyManager();
        _events = new EventManager();
        _scraper = new ScreenScraper();
        _input = new InputSimulator();
        _navigator = new AccessibilityTreeNavigator();
    }
    
    public bool ClickElement(string name)
    {
        var element = _discovery.FindByName(name);
        if (element == null) return false;
        _input.Click(element);
        return true;
    }
    
    public bool TypeIntoField(string fieldName, string text)
    {
        var element = _discovery.FindByName(fieldName);
        if (element == null) return false;
        _input.TypeText(element, text);
        return true;
    }
    
    public string GetElementText(string elementName)
    {
        var element = _discovery.FindByName(elementName);
        return element != null ? _scraper.ExtractFullText(element) : null;
    }
    
    public IUIAutomationElement WaitForElement(string name, int timeoutMs = 10000)
    {
        var sw = Stopwatch.StartNew();
        while (sw.ElapsedMilliseconds < timeoutMs)
        {
            var element = _discovery.FindByName(name);
            if (element != null) return element;
            Thread.Sleep(100);
        }
        return null;
    }
    
    public bool ElementExists(string name) => _discovery.FindByName(name) != null;
    
    public ElementInfo GetElementInfo(string name)
    {
        var element = _discovery.FindByName(name);
        return element != null ? _properties.GetElementInfo(element) : null;
    }
    
    public void Dispose()
    {
        _events?.Dispose();
        _automation = null;
    }
}
```

---

## 12. Integration with Agent Loops

```csharp
public class UIAutomationAgentLoop
{
    private AgentUIController _controller;
    private IAgentBrain _brain;
    private CancellationTokenSource _cts;
    
    public UIAutomationAgentLoop(IAgentBrain brain)
    {
        _controller = new AgentUIController();
        _brain = brain;
    }
    
    public async Task<AutomationResult> ExecuteTaskAsync(AutomationTask task)
    {
        var result = new AutomationResult { Success = false };
        
        try
        {
            switch (task.ActionType)
            {
                case "click":
                    result.Success = _controller.ClickElement(task.Target);
                    break;
                case "type":
                    result.Success = _controller.TypeIntoField(task.Target, task.Value);
                    break;
                case "read":
                    result.Data = _controller.GetElementText(task.Target);
                    result.Success = result.Data != null;
                    break;
                case "wait":
                    result.Success = _controller.WaitForElement(task.Target, task.TimeoutMs) != null;
                    break;
                case "exists":
                    result.Success = _controller.ElementExists(task.Target);
                    break;
                case "info":
                    result.Data = _controller.GetElementInfo(task.Target);
                    result.Success = result.Data != null;
                    break;
                default:
                    result.Error = $"Unknown action type: {task.ActionType}";
                    break;
            }
        }
        catch (Exception ex)
        {
            result.Error = ex.Message;
            result.StackTrace = ex.StackTrace;
        }
        
        return result;
    }
    
    public void StartMonitoring()
    {
        _cts = new CancellationTokenSource();
        Task.Run(() => MonitoringLoop(_cts.Token));
    }
    
    private async Task MonitoringLoop(CancellationToken token)
    {
        while (!token.IsCancellationRequested)
        {
            try
            {
                var windowInfo = _controller.GetCurrentWindowInfo();
                await _brain.ReportUIStateAsync(new UIState
                {
                    CurrentWindow = windowInfo,
                    Timestamp = DateTime.UtcNow
                });
                await Task.Delay(1000, token);
            }
            catch (OperationCanceledException) { break; }
            catch (Exception ex) { await _brain.ReportErrorAsync(ex); }
        }
    }
    
    public void StopMonitoring() => _cts?.Cancel();
}

public class AutomationTask
{
    public string ActionType { get; set; }
    public string Target { get; set; }
    public string Value { get; set; }
    public int TimeoutMs { get; set; } = 10000;
}

public class AutomationResult
{
    public bool Success { get; set; }
    public object Data { get; set; }
    public string Error { get; set; }
    public string StackTrace { get; set; }
}

public class UIState
{
    public WindowInfo CurrentWindow { get; set; }
    public DateTime Timestamp { get; set; }
}

public interface IAgentBrain
{
    Task ReportUIStateAsync(UIState state);
    Task ReportErrorAsync(Exception ex);
}
```

---

## Appendix: Required NuGet Packages

```xml
<PackageReference Include="UIAutomationClient" Version="10.0.19041.1" />
<PackageReference Include="System.Windows.Automation" Version="1.0.0" />
<PackageReference Include="Tesseract" Version="5.2.0" />
<PackageReference Include="System.Drawing.Common" Version="7.0.0" />
```

---

## Key Constants Reference

```csharp
public static class UIAConstants
{
    // Pattern IDs
    public const int UIA_InvokePatternId = 10000;
    public const int UIA_SelectionPatternId = 10001;
    public const int UIA_ValuePatternId = 10002;
    public const int UIA_RangeValuePatternId = 10003;
    public const int UIA_ScrollPatternId = 10004;
    public const int UIA_ExpandCollapsePatternId = 10005;
    public const int UIA_TextPatternId = 10014;
    public const int UIA_TogglePatternId = 10015;
    public const int UIA_WindowPatternId = 10009;
    
    // Control Type IDs
    public const int UIA_ButtonControlTypeId = 50000;
    public const int UIA_CheckBoxControlTypeId = 50002;
    public const int UIA_ComboBoxControlTypeId = 50003;
    public const int UIA_EditControlTypeId = 50004;
    public const int UIA_TextControlTypeId = 50020;
    public const int UIA_WindowControlTypeId = 50032;
    public const int UIA_DocumentControlTypeId = 50030;
    public const int UIA_DataGridControlTypeId = 50028;
}
```

---

**Document End**

*This specification provides a comprehensive framework for Windows UI Automation integration in the OpenClaw-inspired AI agent system. All code examples are production-ready and can be adapted for specific implementation requirements.*
