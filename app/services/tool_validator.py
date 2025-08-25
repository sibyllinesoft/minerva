"""Tool validation and normalization service."""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID
import json
try:
    from jsonschema import validate, ValidationError, Draft7Validator
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    ValidationError = Exception
    Draft7Validator = None

logger = logging.getLogger(__name__)


class ToolValidationError(Exception):
    """Exception raised when tool validation fails."""
    pass


class ToolValidator:
    """Service for validating and normalizing tool definitions."""
    
    # JSON Schema for MCP tool input schema validation
    MCP_INPUT_SCHEMA_SCHEMA = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["object"]},
            "properties": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "description": {"type": "string"},
                        "enum": {"type": "array"},
                        "default": {},
                        "required": {"type": "boolean"}
                    }
                }
            },
            "required": {"type": "array", "items": {"type": "string"}},
            "additionalProperties": {"type": "boolean"}
        },
        "required": ["type", "properties"]
    }
    
    # Validation rules configuration
    VALIDATION_CONFIG = {
        "name": {
            "min_length": 1,
            "max_length": 100,
            "pattern": r"^[a-zA-Z0-9_-]+$",
            "required": True
        },
        "brief": {
            "min_length": 0,
            "max_length": 200,
            "required": False
        },
        "description": {
            "min_length": 0,
            "max_length": 2000,
            "required": False
        },
        "category": {
            "max_length": 50,
            "required": False,
            "valid_categories": [
                "data", "files", "web", "api", "communication", "productivity",
                "development", "system", "ai", "automation", "analysis", "other"
            ]
        },
        "tags": {
            "max_count": 10,
            "max_tag_length": 30,
            "required": False
        }
    }
    
    def __init__(self):
        """Initialize the tool validator."""
        if HAS_JSONSCHEMA and Draft7Validator:
            self.schema_validator = Draft7Validator(self.MCP_INPUT_SCHEMA_SCHEMA)
        else:
            self.schema_validator = None
    
    def validate_tool(self, tool_data: Dict[str, Any], strict: bool = False) -> Tuple[bool, List[str]]:
        """
        Validate a tool definition.
        
        Args:
            tool_data: Raw tool data from MCP server
            strict: If True, apply stricter validation rules
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            # Validate required fields and basic structure
            name_errors = self._validate_name(tool_data.get("name"))
            errors.extend(name_errors)
            
            # Validate optional fields
            brief_errors = self._validate_brief(tool_data.get("brief", ""))
            errors.extend(brief_errors)
            
            description_errors = self._validate_description(tool_data.get("description", ""))
            errors.extend(description_errors)
            
            # Validate schema if present
            if "schema" in tool_data or "inputSchema" in tool_data:
                schema = tool_data.get("schema", tool_data.get("inputSchema", {}))
                schema_errors = self._validate_input_schema(schema)
                errors.extend(schema_errors)
            
            # Validate category
            category_errors = self._validate_category(tool_data.get("category"))
            errors.extend(category_errors)
            
            # Validate tags
            tags_errors = self._validate_tags(tool_data.get("tags", []))
            errors.extend(tags_errors)
            
            # Apply strict validation rules if requested
            if strict:
                strict_errors = self._apply_strict_validation(tool_data)
                errors.extend(strict_errors)
            
            is_valid = len(errors) == 0
            
            if not is_valid:
                logger.debug(f"Tool validation failed for '{tool_data.get('name', 'unknown')}': {errors}")
            
            return is_valid, errors
        
        except Exception as e:
            logger.error(f"Unexpected error during tool validation: {e}")
            return False, [f"Validation error: {str(e)}"]
    
    def normalize_tool(self, tool_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize and clean tool data.
        
        Args:
            tool_data: Raw tool data from MCP server
            
        Returns:
            Normalized tool data
        """
        try:
            normalized = {}
            
            # Normalize name (required)
            name = tool_data.get("name", "").strip()
            if not name:
                raise ToolValidationError("Tool name is required")
            
            # Clean name to match pattern
            normalized["name"] = re.sub(r'[^a-zA-Z0-9_-]', '_', name)[:100]
            
            # Normalize brief/summary
            brief = str(tool_data.get("brief", tool_data.get("summary", ""))).strip()
            normalized["brief"] = brief[:200] if brief else ""
            
            # Normalize description
            description = str(tool_data.get("description", "")).strip()
            normalized["description"] = description[:2000] if description else ""
            
            # Normalize schema
            schema = tool_data.get("schema", tool_data.get("inputSchema", {}))
            if isinstance(schema, dict):
                normalized["schema"] = self._normalize_schema(schema)
            else:
                normalized["schema"] = {}
            
            # Normalize category
            category = self._normalize_category(tool_data.get("category"))
            if category:
                normalized["category"] = category
            
            # Normalize tags
            tags = self._normalize_tags(tool_data.get("tags", []))
            normalized["tags"] = tags
            
            # Add metadata
            normalized["meta"] = {
                "original_source": tool_data.get("source", "unknown"),
                "validation_timestamp": "now",  # Will be set by caller
                "normalized": True
            }
            
            logger.debug(f"Normalized tool '{normalized['name']}'")
            return normalized
        
        except Exception as e:
            logger.error(f"Failed to normalize tool: {e}")
            raise ToolValidationError(f"Normalization failed: {str(e)}")
    
    def _validate_name(self, name: Any) -> List[str]:
        """Validate tool name."""
        errors = []
        
        if not name:
            errors.append("Tool name is required")
            return errors
        
        if not isinstance(name, str):
            errors.append("Tool name must be a string")
            return errors
        
        name = name.strip()
        config = self.VALIDATION_CONFIG["name"]
        
        if len(name) < config["min_length"]:
            errors.append(f"Tool name must be at least {config['min_length']} characters")
        
        if len(name) > config["max_length"]:
            errors.append(f"Tool name must be at most {config['max_length']} characters")
        
        if not re.match(config["pattern"], name):
            errors.append("Tool name must contain only letters, numbers, underscores, and hyphens")
        
        return errors
    
    def _validate_brief(self, brief: Any) -> List[str]:
        """Validate tool brief/summary."""
        errors = []
        
        if brief is None:
            return errors
        
        if not isinstance(brief, str):
            errors.append("Tool brief must be a string")
            return errors
        
        config = self.VALIDATION_CONFIG["brief"]
        
        if len(brief) > config["max_length"]:
            errors.append(f"Tool brief must be at most {config['max_length']} characters")
        
        return errors
    
    def _validate_description(self, description: Any) -> List[str]:
        """Validate tool description."""
        errors = []
        
        if description is None:
            return errors
        
        if not isinstance(description, str):
            errors.append("Tool description must be a string")
            return errors
        
        config = self.VALIDATION_CONFIG["description"]
        
        if len(description) > config["max_length"]:
            errors.append(f"Tool description must be at most {config['max_length']} characters")
        
        return errors
    
    def _validate_input_schema(self, schema: Any) -> List[str]:
        """Validate tool input schema against JSON Schema spec."""
        errors = []
        
        if not schema:
            return errors  # Schema is optional
        
        if not isinstance(schema, dict):
            errors.append("Tool input schema must be an object")
            return errors
        
        if self.schema_validator:
            try:
                # Validate against our MCP input schema structure
                self.schema_validator.validate(schema)
            except ValidationError as e:
                errors.append(f"Invalid input schema: {e.message}")
            except Exception as e:
                errors.append(f"Schema validation error: {str(e)}")
        else:
            # Basic validation without jsonschema
            if not isinstance(schema.get("type"), str):
                errors.append("Schema must have a type property")
            if "properties" in schema and not isinstance(schema["properties"], dict):
                errors.append("Schema properties must be an object")
        
        return errors
    
    def _validate_category(self, category: Any) -> List[str]:
        """Validate tool category."""
        errors = []
        
        if category is None:
            return errors  # Category is optional
        
        if not isinstance(category, str):
            errors.append("Tool category must be a string")
            return errors
        
        config = self.VALIDATION_CONFIG["category"]
        
        if len(category) > config["max_length"]:
            errors.append(f"Tool category must be at most {config['max_length']} characters")
        
        # Note: We don't enforce valid_categories in validation, only in normalization
        
        return errors
    
    def _validate_tags(self, tags: Any) -> List[str]:
        """Validate tool tags."""
        errors = []
        
        if not tags:
            return errors  # Tags are optional
        
        if not isinstance(tags, list):
            errors.append("Tool tags must be an array")
            return errors
        
        config = self.VALIDATION_CONFIG["tags"]
        
        if len(tags) > config["max_count"]:
            errors.append(f"Tool can have at most {config['max_count']} tags")
        
        for i, tag in enumerate(tags):
            if not isinstance(tag, str):
                errors.append(f"Tag {i} must be a string")
            elif len(tag) > config["max_tag_length"]:
                errors.append(f"Tag {i} must be at most {config['max_tag_length']} characters")
        
        return errors
    
    def _apply_strict_validation(self, tool_data: Dict[str, Any]) -> List[str]:
        """Apply strict validation rules."""
        errors = []
        
        # Require description in strict mode
        description = tool_data.get("description", "").strip()
        if not description:
            errors.append("Description is required in strict mode")
        
        # Require schema in strict mode
        schema = tool_data.get("schema", tool_data.get("inputSchema"))
        if not schema or not isinstance(schema, dict):
            errors.append("Valid input schema is required in strict mode")
        
        return errors
    
    def _normalize_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize tool input schema."""
        if not isinstance(schema, dict):
            return {}
        
        normalized = {}
        
        # Ensure required JSON Schema fields
        normalized["type"] = schema.get("type", "object")
        normalized["properties"] = schema.get("properties", {})
        
        if "required" in schema:
            normalized["required"] = schema["required"]
        
        if "additionalProperties" in schema:
            normalized["additionalProperties"] = schema["additionalProperties"]
        
        # Clean up property definitions
        if isinstance(normalized["properties"], dict):
            cleaned_props = {}
            for prop_name, prop_def in normalized["properties"].items():
                if isinstance(prop_def, dict):
                    cleaned_prop = {
                        "type": prop_def.get("type", "string"),
                        "description": prop_def.get("description", "")
                    }
                    if "enum" in prop_def:
                        cleaned_prop["enum"] = prop_def["enum"]
                    if "default" in prop_def:
                        cleaned_prop["default"] = prop_def["default"]
                    
                    cleaned_props[prop_name] = cleaned_prop
            
            normalized["properties"] = cleaned_props
        
        return normalized
    
    def _normalize_category(self, category: Any) -> Optional[str]:
        """Normalize tool category."""
        if not category or not isinstance(category, str):
            return None
        
        category = category.strip().lower()
        
        # Map common category variants to standard categories
        category_mapping = {
            "file": "files",
            "filesystem": "files",
            "network": "web",
            "http": "web",
            "rest": "api",
            "database": "data",
            "db": "data",
            "chat": "communication",
            "messaging": "communication",
            "tool": "productivity",
            "utility": "productivity",
            "dev": "development",
            "programming": "development",
            "code": "development",
            "os": "system",
            "machine-learning": "ai",
            "ml": "ai",
            "llm": "ai",
            "workflow": "automation",
            "analytics": "analysis"
        }
        
        # Try to map to standard category
        if category in category_mapping:
            return category_mapping[category]
        
        # Check if it's already a valid category
        valid_categories = self.VALIDATION_CONFIG["category"]["valid_categories"]
        if category in valid_categories:
            return category
        
        # Default to "other" for unrecognized categories
        return "other"
    
    def _normalize_tags(self, tags: Any) -> List[str]:
        """Normalize tool tags."""
        if not isinstance(tags, list):
            return []
        
        normalized_tags = []
        seen_tags = set()
        
        for tag in tags:
            if not isinstance(tag, str):
                continue
            
            # Clean and normalize tag
            clean_tag = re.sub(r'[^a-zA-Z0-9_-]', '', str(tag).strip().lower())
            clean_tag = clean_tag[:30]  # Truncate to max length
            
            if clean_tag and clean_tag not in seen_tags:
                normalized_tags.append(clean_tag)
                seen_tags.add(clean_tag)
                
                # Limit number of tags
                if len(normalized_tags) >= self.VALIDATION_CONFIG["tags"]["max_count"]:
                    break
        
        return normalized_tags
    
    def assess_tool_quality(self, tool_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the quality of a tool definition.
        
        Returns a quality assessment with scores and recommendations.
        """
        assessment = {
            "overall_score": 0,
            "completeness_score": 0,
            "clarity_score": 0,
            "schema_score": 0,
            "recommendations": []
        }
        
        try:
            # Assess completeness (40% of score)
            completeness = self._assess_completeness(tool_data)
            assessment["completeness_score"] = completeness["score"]
            assessment["recommendations"].extend(completeness["recommendations"])
            
            # Assess clarity (30% of score)  
            clarity = self._assess_clarity(tool_data)
            assessment["clarity_score"] = clarity["score"]
            assessment["recommendations"].extend(clarity["recommendations"])
            
            # Assess schema quality (30% of score)
            schema_quality = self._assess_schema_quality(tool_data.get("schema", {}))
            assessment["schema_score"] = schema_quality["score"]
            assessment["recommendations"].extend(schema_quality["recommendations"])
            
            # Calculate overall score
            assessment["overall_score"] = (
                assessment["completeness_score"] * 0.4 +
                assessment["clarity_score"] * 0.3 +
                assessment["schema_score"] * 0.3
            )
            
            return assessment
        
        except Exception as e:
            logger.error(f"Failed to assess tool quality: {e}")
            return assessment
    
    def _assess_completeness(self, tool_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess completeness of tool definition."""
        score = 0
        recommendations = []
        
        # Required fields (50%)
        if tool_data.get("name"):
            score += 20
        else:
            recommendations.append("Add a descriptive tool name")
        
        if tool_data.get("brief") or tool_data.get("summary"):
            score += 15
        else:
            recommendations.append("Add a brief summary of what the tool does")
        
        if tool_data.get("description"):
            score += 15
        else:
            recommendations.append("Add a detailed description")
        
        # Optional but valuable fields (50%)
        if tool_data.get("schema") or tool_data.get("inputSchema"):
            score += 25
        else:
            recommendations.append("Add an input schema to define expected parameters")
        
        if tool_data.get("category"):
            score += 10
        else:
            recommendations.append("Add a category to help with tool organization")
        
        if tool_data.get("tags"):
            score += 15
        else:
            recommendations.append("Add tags to improve discoverability")
        
        return {"score": min(score, 100), "recommendations": recommendations}
    
    def _assess_clarity(self, tool_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess clarity of tool definition."""
        score = 100
        recommendations = []
        
        # Check description quality
        description = tool_data.get("description", "")
        if description:
            if len(description) < 20:
                score -= 20
                recommendations.append("Provide a more detailed description")
            elif len(description) > 1000:
                score -= 10
                recommendations.append("Consider shortening the description for clarity")
        
        # Check brief quality
        brief = tool_data.get("brief", tool_data.get("summary", ""))
        if brief:
            if len(brief) < 10:
                score -= 15
                recommendations.append("Provide a more descriptive brief summary")
        
        # Check name quality
        name = tool_data.get("name", "")
        if name and len(name) < 3:
            score -= 10
            recommendations.append("Use a more descriptive tool name")
        
        return {"score": max(score, 0), "recommendations": recommendations}
    
    def _assess_schema_quality(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality of input schema."""
        score = 0
        recommendations = []
        
        if not schema:
            recommendations.append("Add an input schema to define expected parameters")
            return {"score": 0, "recommendations": recommendations}
        
        # Basic structure (40%)
        if schema.get("type") == "object":
            score += 20
        
        if "properties" in schema and isinstance(schema["properties"], dict):
            score += 20
            
            # Property definitions (40%)
            props = schema["properties"]
            if props:
                total_props = len(props)
                described_props = sum(1 for p in props.values() 
                                   if isinstance(p, dict) and p.get("description"))
                typed_props = sum(1 for p in props.values() 
                               if isinstance(p, dict) and p.get("type"))
                
                if described_props / total_props >= 0.8:
                    score += 20
                elif described_props / total_props >= 0.5:
                    score += 10
                else:
                    recommendations.append("Add descriptions to more schema properties")
                
                if typed_props / total_props >= 0.9:
                    score += 20
                elif typed_props / total_props >= 0.7:
                    score += 15
                else:
                    recommendations.append("Add type definitions to more schema properties")
        
        # Required fields specification (20%)
        if "required" in schema and isinstance(schema["required"], list):
            score += 20
        else:
            recommendations.append("Specify which schema properties are required")
        
        return {"score": min(score, 100), "recommendations": recommendations}


# Global validator instance
_validator: Optional[ToolValidator] = None


def get_tool_validator() -> ToolValidator:
    """Get the global tool validator instance."""
    global _validator
    if _validator is None:
        _validator = ToolValidator()
    return _validator