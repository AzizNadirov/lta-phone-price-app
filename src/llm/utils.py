import json
import re
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, ValidationError
from loguru import logger

try:
    import litellm
except ImportError:
    litellm = None
    logger.warning("litellm not available. Auto-completion features will be disabled.")


class ResponseSchema(BaseModel):
    """
    A schema class for defining structure and extracting data from raw LLM responses.
    
    Usage:
        class PersonSchema(ResponseSchema):
            name: str = Field(default="", description="Person's full name")
            age: int = Field(default=0, description="Person's age in years")
            email: Optional[str] = Field(default=None, description="Email address")
        
        schema = PersonSchema()
        result = schema.extract_from("John Doe is 25 years old, email: john@example.com")
    """
    
    model_config = {"arbitrary_types_allowed": True}
    
    def extract_from(self, result: str, auto_complete: bool = False, model: str = None) -> Dict[str, Any]:
        """
        Extract structured data from raw LLM text response.
        
        Args:
            result (str): Raw text response from LLM
            auto_complete (bool): Whether to use LLM completion for extraction assistance
            model (str): Model identifier for litellm.completion (e.g., "gemini/gemini-2.0-flash")
        
        Returns:
            Dict[str, Any]: Extracted structured data matching the schema
        
        Raises:
            ValidationError: If extracted data doesn't match schema
            ValueError: If extraction fails
        """
        try:
            # First, try to extract JSON if present
            extracted_data = self._extract_json(result)
            
            if not extracted_data:
                # Try pattern-based extraction
                extracted_data = self._extract_patterns(result)
            
            if not extracted_data and auto_complete and model and litellm:
                # Use LLM to assist with extraction
                extracted_data = self._llm_assisted_extraction(result, model)
            
            if not extracted_data:
                # Fallback to field-by-field extraction
                extracted_data = self._extract_fields(result)
            
            # Validate against schema
            validated_data = self.model_validate(extracted_data)
            return validated_data.model_dump()
            
        except ValidationError as e:
            if auto_complete and model and litellm:
                # Try LLM-assisted correction
                try:
                    corrected_data = self._llm_correction(result, extracted_data, str(e), model)
                    return self.model_validate(corrected_data).model_dump()
                except:
                    pass
            raise ValueError(f"Failed to extract valid data: {e}")
    
    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON objects from text."""
        json_patterns = [
            r'\{[^{}]*\}',  # Simple JSON object
            r'\{.*?\}',     # JSON with nested objects (non-greedy)
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        return None
    
    def _extract_patterns(self, text: str) -> Dict[str, Any]:
        """Extract data using common patterns."""
        data = {}
        
        # Get field information from schema
        field_info = self._get_field_info()
        
        for field_name, field_data in field_info.items():
            field_type = field_data['type']
            description = field_data.get('description', field_name)
            
            # Try various extraction patterns
            patterns = self._generate_patterns(field_name, field_type, description)
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    try:
                        value = self._convert_value(match.group(1), field_type)
                        data[field_name] = value
                        break
                    except (ValueError, IndexError):
                        continue
        
        return data
    
    def _generate_patterns(self, field_name: str, field_type: Type, description: str) -> List[str]:
        """Generate regex patterns for field extraction."""
        patterns = []
        
        # Common separators and indicators
        separators = [r':', r'=', r'is', r'was', r'are', r'were', r'-']
        
        # Pattern variations for field name
        field_variations = [
            field_name.lower(),
            field_name.replace('_', ' '),
            field_name.replace('_', '-'),
            description.lower() if description else field_name
        ]
        
        for variation in field_variations:
            for sep in separators:
                if field_type in [int, float]:
                    # Numeric patterns
                    patterns.append(rf'{re.escape(variation)}\s*{sep}\s*(\d+(?:\.\d+)?)')
                elif field_type == bool:
                    # Boolean patterns
                    patterns.append(rf'{re.escape(variation)}\s*{sep}\s*(true|false|yes|no)', )
                else:
                    # String patterns
                    patterns.extend([
                        rf'{re.escape(variation)}\s*{sep}\s*["\']([^"\']+)["\']',  # Quoted
                        rf'{re.escape(variation)}\s*{sep}\s*([^\n,;]+)',           # Unquoted
                    ])
        
        return patterns
    
    def _extract_fields(self, text: str) -> Dict[str, Any]:
        """Fallback field extraction using keywords and context."""
        data = {}
        field_info = self._get_field_info()
        
        for field_name, field_data in field_info.items():
            field_type = field_data['type']
            
            # Simple keyword-based extraction
            if field_type == str:
                # Look for quoted strings or capitalized words
                if field_name.lower() in ['name', 'title']:
                    pattern = r'([A-Z][a-z]+ [A-Z][a-z]+|[A-Z][a-z]+)'
                    matches = re.findall(pattern, text)
                    if matches:
                        data[field_name] = matches[0]
                elif field_name.lower() in ['email']:
                    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                    matches = re.findall(pattern, text)
                    if matches:
                        data[field_name] = matches[0]
            
            elif field_type in [int, float]:
                # Look for numbers in context
                numbers = re.findall(r'\d+(?:\.\d+)?', text)
                if numbers:
                    try:
                        data[field_name] = field_type(numbers[0])
                    except ValueError:
                        pass
        
        return data
    
    def _llm_assisted_extraction(self, text: str, model: str) -> Dict[str, Any]:
        """Use LLM to assist with data extraction."""
        if not litellm:
            return {}
        
        schema_info = self._get_schema_description()
        
        prompt = f"""
        Extract structured data from the following text according to this schema:
        
        Schema: {schema_info}
        
        Text: {text}
        
        Please return only a valid JSON object with the extracted data. If a field cannot be found, use null.
        """
        
        try:
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content
            return self._extract_json(response_text) or {}
            
        except Exception as e:
            logger.warning(f"LLM-assisted extraction failed: {e}")
            return {}
    
    def _llm_correction(self, original_text: str, extracted_data: Dict, error: str, model: str) -> Dict[str, Any]:
        """Use LLM to correct extraction errors."""
        if not litellm:
            raise ValueError("LLM correction not available")
        
        schema_info = self._get_schema_description()
        
        prompt = f"""
        Fix the extracted data to match the required schema:
        
        Schema: {schema_info}
        Original text: {original_text}
        Extracted data: {json.dumps(extracted_data)}
        Validation error: {error}
        
        Please return only a corrected JSON object that matches the schema exactly.
        """
        
        try:
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content
            corrected = self._extract_json(response_text)
            if corrected:
                return corrected
            
        except Exception as e:
            logger.warning(f"LLM correction failed: {e}")
        
        raise ValueError("Could not correct extraction errors")
    
    def _get_field_info(self) -> Dict[str, Dict[str, Any]]:
        """Get field information from the schema."""
        field_info = {}
        
        for field_name, field in self.__class__.model_fields.items():
            field_info[field_name] = {
                'type': field.annotation,
                'description': field.description,
                'default': field.default,
                'required': field.is_required()
            }
        
        return field_info
    
    def _get_schema_description(self) -> str:
        """Get a human-readable schema description."""
        field_info = self._get_field_info()
        descriptions = []
        
        for field_name, info in field_info.items():
            type_name = getattr(info['type'], '__name__', str(info['type']))
            desc = info.get('description', '')
            required = "required" if info['required'] else "optional"
            
            descriptions.append(f"- {field_name} ({type_name}, {required}): {desc}")
        
        return "\n".join(descriptions)
    
    def _convert_value(self, value: str, target_type: Type) -> Any:
        """Convert string value to target type."""
        if target_type == str:
            return value.strip()
        elif target_type == int:
            return int(float(value))  # Handle "25.0" -> 25
        elif target_type == float:
            return float(value)
        elif target_type == bool:
            return value.lower() in ['true', 'yes', '1', 'on']
        else:
            return value
