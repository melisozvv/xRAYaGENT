#!/usr/bin/env python3
"""
Test script to demonstrate JSON parsing functionality from MedGemma responses
"""

import json
import re
from typing import Dict, Any, List, Union

def extract_and_parse_json(text: str) -> Union[Dict[str, Any], List[Any], str]:
    """
    Extract JSON from markdown code blocks and parse it into actual JSON objects.
    
    Args:
        text: Raw text that may contain JSON in markdown code blocks
        
    Returns:
        Parsed JSON object if valid JSON found, otherwise original text
    """
    # Pattern to match ```json ... ``` blocks
    json_pattern = r'```json\s*\n(.*?)\n```'
    
    # Try to find JSON in code blocks
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    if matches:
        # Take the first JSON block found
        json_str = matches[0].strip()
        try:
            # Parse the JSON
            parsed_json = json.loads(json_str)
            print(f"✅ Successfully parsed JSON: {parsed_json}")
            return parsed_json
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON block: {e}")
            print(f"JSON content: {json_str}")
    
    # If no JSON blocks found, try to extract JSON from the entire text
    # Look for patterns that start with { or [ (common JSON starts)
    json_pattern_loose = r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])'
    
    matches = re.findall(json_pattern_loose, text)
    if matches:
        for match in matches:
            try:
                parsed_json = json.loads(match.strip())
                print(f"✅ Successfully parsed loose JSON: {parsed_json}")
                return parsed_json
            except json.JSONDecodeError:
                continue
    
    # If no valid JSON found, return original text
    print(f"ℹ️  No valid JSON found, returning original text")
    return text

def test_json_parsing():
    """Test the JSON parsing with various input formats"""
    
    test_cases = [
        {
            "name": "Markdown JSON block (your example)",
            "input": "```json\n{\n  \"EXIST\": 1,\n  \"DISTANCE\": 4.5\n}\n```",
            "expected_type": dict
        },
        {
            "name": "JSON with surrounding text",
            "input": "Based on the chest X-ray analysis:\n\n```json\n{\n  \"pneumonia\": \"present\",\n  \"confidence\": 0.85,\n  \"location\": \"left lower lobe\"\n}\n```\n\nThis indicates pneumonia in the left lower lobe.",
            "expected_type": dict
        },
        {
            "name": "Malformed JSON",
            "input": "```json\n{\n  \"EXIST\": 1,\n  \"DISTANCE\": 4.5\n  missing_comma: true\n}\n```",
            "expected_type": str
        },
        {
            "name": "Plain text response",
            "input": "No evidence of pneumonia is visible in this chest X-ray. The lungs appear clear and well-expanded.",
            "expected_type": str
        },
        {
            "name": "JSON array",
            "input": "```json\n[\n  {\"finding\": \"consolidation\", \"location\": \"left lung\"},\n  {\"finding\": \"pleural_effusion\", \"location\": \"right lung\"}\n]\n```",
            "expected_type": list
        },
        {
            "name": "Inline JSON without markdown",
            "input": "The analysis results are: {\"normal\": true, \"abnormalities\": []} and no further action needed.",
            "expected_type": dict
        }
    ]
    
    print("🧪 Testing JSON Parsing Functionality")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 40)
        print(f"Input: {test_case['input'][:100]}{'...' if len(test_case['input']) > 100 else ''}")
        
        result = extract_and_parse_json(test_case['input'])
        result_type = type(result)
        expected_type = test_case['expected_type']
        
        print(f"Expected type: {expected_type.__name__}")
        print(f"Actual type: {result_type.__name__}")
        print(f"Result: {result}")
        
        if result_type == expected_type:
            print("✅ PASS")
        else:
            print("❌ FAIL")

def demonstrate_usage():
    """Demonstrate how this would work in the main processing"""
    
    print("\n" + "=" * 60)
    print("📋 Demonstration: How it works in main processing")
    print("=" * 60)
    
    # Simulate MedGemma response
    raw_medgemma_response = """Based on my analysis of this chest X-ray, here are the findings:

```json
{
  "pneumonia_present": true,
  "confidence_score": 0.87,
  "location": "right lower lobe",
  "severity": "moderate",
  "additional_findings": ["mild cardiomegaly"],
  "recommendation": "antibiotic treatment recommended"
}
```

The consolidation pattern in the right lower lobe is consistent with bacterial pneumonia."""

    print("Raw MedGemma Response:")
    print(raw_medgemma_response)
    print("\n" + "-" * 40)
    
    # Parse it
    parsed_answer = extract_and_parse_json(raw_medgemma_response)
    
    print("Parsed Answer:")
    print(json.dumps(parsed_answer, indent=2) if isinstance(parsed_answer, (dict, list)) else parsed_answer)
    
    print(f"\nAnswer Type: {type(parsed_answer).__name__}")
    
    # Show how it would be stored
    sample_result = {
        "sample_id": "example_001",
        "question": "Is there evidence of pneumonia in this chest X-ray?",
        "answer": parsed_answer,  # This is now actual JSON!
        "raw_answer": raw_medgemma_response,
        "answer_type": type(parsed_answer).__name__
    }
    
    print("\n" + "-" * 40)
    print("Final Sample Result Structure:")
    print(json.dumps(sample_result, indent=2, default=str))

if __name__ == "__main__":
    test_json_parsing()
    demonstrate_usage()
    print("\n🎉 JSON parsing test completed!") 