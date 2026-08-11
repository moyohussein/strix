#!/usr/bin/env python3
"""
Test script to verify if a local model can handle structured tool calls required by Strix.

This script tests whether your local inference server can properly return structured
`tool_calls` instead of plain text, which is essential for Strix's agentic capabilities.

Usage:
    python scripts/test_local_model_tool_call.py --model "your-model" --api-base "http://localhost:11434"

Examples:
    # Test Ollama model
    python scripts/test_local_model_tool_call.py --model "qwen3:70b" --api-base "http://localhost:11434"

    # Test custom OpenAI-compatible server
    python scripts/test_local_model_tool_call.py --model "local-model" --api-base "http://localhost:8000/v1"
"""

import argparse
import json
import sys
from typing import Any

try:
    import requests
except ImportError:
    print("Error: requests library is required. Install with: pip install requests")
    sys.exit(1)


class ToolCallTester:
    """Test a model's ability to return structured tool calls."""

    def __init__(self, model: str, api_base: str, api_key: str | None = None):
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def _get_endpoint(self) -> str:
        """Get the appropriate endpoint based on API base."""
        if self.api_base.endswith("/v1") or "/v1/" in self.api_base:
            return f"{self.api_base}/chat/completions"
        elif self.api_base.endswith("/api"):
            return f"{self.api_base}/chat/completions"
        else:
            # Assume it's a base URL without /v1
            return f"{self.api_base}/v1/chat/completions"

    def test_structured_tool_calls(self) -> dict[str, Any]:
        """Test if the model returns structured tool calls."""
        endpoint = self._get_endpoint()
        
        # Test payload with tools
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant with access to tools. Always use tools when appropriate."
                },
                {
                    "role": "user",
                    "content": "List all files in the current directory using the available tool."
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "list_files",
                    "description": "List files in a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The directory path to list"
                            }
                        },
                        "required": ["path"]
                    }
                },
                {
                    "type": "function",
                    "name": "exec_command",
                    "description": "Execute a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cmd": {"type": "string", "description": "Command to execute"}
                        },
                        "required": ["cmd"]
                    }
                }
            ],
            "tool_choice": "auto",
            "temperature": 0.2,
            "max_tokens": 1000
        }

        try:
            response = requests.post(
                endpoint,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return self._analyze_response(response.json())
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Request failed: {str(e)}",
                "endpoint": endpoint,
                "model": self.model
            }

    def _analyze_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Analyze the model response for tool call capabilities."""
        result: dict[str, Any] = {
            "success": True,
            "model": self.model,
            "api_base": self.api_base,
            "structured_tool_calls": False,
            "tool_calls_found": False,
            "tool_call_format": None,
            "text_content_present": False,
            "issues": []
        }

        # Check if we got choices
        if "choices" not in response or len(response["choices"]) == 0:
            result["success"] = False
            result["issues"].append("No choices in response")
            return result

        choice = response["choices"][0]
        message = choice.get("message", {})
        
        # Check for structured tool calls
        if "tool_calls" in message and message["tool_calls"]:
            result["structured_tool_calls"] = True
            result["tool_calls_found"] = True
            result["tool_call_format"] = "structured"
            
            # Verify tool call structure
            for i, tool_call in enumerate(message["tool_calls"]):
                if not isinstance(tool_call, dict):
                    result["issues"].append(f"Tool call {i} is not a dict: {type(tool_call)}")
                elif "type" not in tool_call or tool_call["type"] != "function":
                    result["issues"].append(f"Tool call {i} missing or invalid type")
                elif "function" not in tool_call:
                    result["issues"].append(f"Tool call {i} missing function field")
                elif "name" not in tool_call["function"]:
                    result["issues"].append(f"Tool call {i} missing function name")
            
            if result["issues"]:
                result["success"] = False
            
        # Check for text content that might contain tool calls
        elif "content" in message and message["content"]:
            content = message["content"]
            result["text_content_present"] = True
            
            # Look for common patterns that indicate tool calls in text
            content_lower = content.lower()
            if any(pattern in content_lower for pattern in [
                "tool_call", "exec_command", "list_files", 
                "<tool>", "<function", "function:", 
                "{\"name\":", "{\"action\":"
            ]):
                result["tool_call_format"] = "text_form"
                result["issues"].append(
                    "Model returns tool calls as text content instead of structured tool_calls field. "
                    "This will NOT work with Strix."
                )
                result["success"] = False
        else:
            result["issues"].append("No tool calls or content found in response")
            result["success"] = False

        return result

    def test_context_window(self, target_tokens: int = 16000) -> dict[str, Any]:
        """Test if the model can handle a large context window."""
        endpoint = self._get_endpoint()
        
        # Create a large prompt that simulates Strix's context
        large_context = """
You are a security testing assistant with access to the following tools:

""" + "\n".join([
            f"- Tool {i}: Description of tool {i} with parameters and usage instructions. "
            f"This tool helps with security testing and has various configuration options."
            for i in range(100)
        ])

        # Add user message
        large_context += """

User request: Perform a security analysis using the appropriate tools.
"""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": large_context},
                {"role": "user", "content": "What tools are available for security testing?"}
            ],
            "max_tokens": 100,
            "temperature": 0.2
        }

        try:
            response = requests.post(
                endpoint,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            # If we get a response without context truncation errors, it passed
            return {
                "success": True,
                "model": self.model,
                "context_window_test": True,
                "target_tokens": target_tokens,
                "message": f"Model accepted {len(large_context)} token context successfully"
            }
            
        except requests.exceptions.RequestException as e:
            error_str = str(e).lower()
            if "context" in error_str or "too many tokens" in error_str:
                return {
                    "success": False,
                    "model": self.model,
                    "context_window_test": False,
                    "target_tokens": target_tokens,
                    "error": f"Context window too small: {str(e)}",
                    "message": f"Model cannot handle {target_tokens} token context"
                }
            else:
                return {
                    "success": False,
                    "model": self.model,
                    "context_window_test": False,
                    "error": f"Request failed: {str(e)}"
                }

    def get_model_info(self) -> dict[str, Any]:
        """Get basic model information."""
        endpoint = self._get_endpoint()
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "What is your name and version?"}],
            "max_tokens": 50,
            "temperature": 0.0
        }

        try:
            response = requests.post(
                endpoint,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            choice = response.json()["choices"][0]
            content = choice["message"].get("content", "")
            
            return {
                "model": self.model,
                "response_sample": content[:200] + "..." if len(content) > 200 else content,
                "model_id": response.json().get("model", self.model)
            }
            
        except Exception as e:
            return {
                "model": self.model,
                "error": str(e)
            }


def print_test_results(results: dict[str, Any], test_type: str = "Tool Call Test") -> None:
    """Print formatted test results."""
    print(f"\n{'='*60}")
    print(f"{test_type}")
    print(f"{'='*60}")
    
    if not results.get("success", False):
        print("❌ FAILED")
        if "error" in results:
            print(f"Error: {results['error']}")
        if "issues" in results and results["issues"]:
            print("Issues found:")
            for issue in results["issues"]:
                print(f"  - {issue}")
    else:
        print("✅ PASSED")
        
        if "structured_tool_calls" in results:
            if results["structured_tool_calls"]:
                print("✅ Model returns structured tool calls")
            else:
                print("❌ Model does NOT return structured tool calls")
                
        if "tool_call_format" in results:
            print(f"Tool call format: {results['tool_call_format']}")
            
        if "tool_calls_found" in results:
            print(f"Tool calls found: {results['tool_calls_found']}")
            
        if "text_content_present" in results and results["text_content_present"]:
            print("⚠️  Model returned text content (tool calls may be in text form)")
            
        if "context_window_test" in results:
            if results["context_window_test"]:
                print(f"✅ Context window test passed: {results.get('message', 'OK')}")
            else:
                print(f"❌ Context window test failed: {results.get('error', 'Unknown error')}")
    
    print(f"Model: {results.get('model', 'Unknown')}")
    if "api_base" in results:
        print(f"API Base: {results['api_base']}")


def print_model_capability_summary(model: str, api_base: str, tool_test: dict[str, Any], 
                                  context_test: dict[str, Any]) -> None:
    """Print a comprehensive capability summary."""
    print(f"\n{'='*60}")
    print("MODEL CAPABILITY SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"API Base: {api_base}")
    print()
    
    # Tool call capability
    tool_ok = tool_test.get("success", False) and tool_test.get("structured_tool_calls", False)
    if tool_ok:
        print("✅ Tool Call Capability: PASSED")
        print("   Model can return structured tool_calls field")
    else:
        print("❌ Tool Call Capability: FAILED")
        if "issues" in tool_test:
            for issue in tool_test["issues"][:2]:  # Show first 2 issues
                print(f"   - {issue}")
    
    print()
    
    # Context window capability
    context_ok = context_test.get("success", False)
    if context_ok:
        print(f"✅ Context Window: PASSED")
        print(f"   Model can handle large contexts (≥16k tokens)")
    else:
        print("❌ Context Window: FAILED")
        print(f"   {context_test.get('error', 'Test failed')}")
    
    print()
    
    # Overall recommendation
    if tool_ok and context_ok:
        print("🎉 OVERALL: MODEL IS COMPATIBLE WITH STRIX")
        print("   This model should work well with Strix for pentesting tasks.")
    else:
        print("⚠️  OVERALL: MODEL MAY NOT WORK WITH STRIX")
        print("   Issues must be resolved before using with Strix.")
        if not tool_ok:
            print("   - Structured tool calls are required")
        if not context_ok:
            print("   - Insufficient context window")


def main():
    parser = argparse.ArgumentParser(
        description="Test if a local model can handle structured tool calls for Strix"
    )
    parser.add_argument(
        "--model", "-m", 
        required=True,
        help="Model name/ID to test (e.g., 'qwen3:70b', 'llama4:70b')"
    )
    parser.add_argument(
        "--api-base", "-b",
        required=True,
        help="API base URL (e.g., 'http://localhost:11434' for Ollama)"
    )
    parser.add_argument(
        "--api-key", "-k",
        default=None,
        help="API key if required (not needed for most local models)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only show summary, not detailed test results"
    )
    parser.add_argument(
        "--context-test",
        action="store_true",
        default=True,
        help="Test context window capability (default: True)"
    )

    args = parser.parse_args()

    # Create tester
    tester = ToolCallTester(args.model, args.api_base, args.api_key)

    print(f"Testing model: {args.model}")
    print(f"API base: {args.api_base}")
    print()

    # Run tests
    tool_test = tester.test_structured_tool_calls()
    
    if not args.quiet:
        print_test_results(tool_test, "Structured Tool Call Test")
    
    # Context window test
    context_test = None
    if args.context_test:
        context_test = tester.test_context_window()
        if not args.quiet:
            print_test_results(context_test, "Context Window Test")
    
    # Get model info
    model_info = tester.get_model_info()
    if not args.quiet and "error" not in model_info:
        print(f"\nModel Info:")
        print(f"  Sample response: {model_info.get('response_sample', 'N/A')}")
    
    # Print summary
    if context_test:
        print_model_capability_summary(args.model, args.api_base, tool_test, context_test)
    else:
        print_model_capability_summary(args.model, args.api_base, tool_test, {"success": False, "error": "Skipped"})

    # Exit with appropriate code
    tool_ok = tool_test.get("success", False) and tool_test.get("structured_tool_calls", False)
    context_ok = context_test.get("success", False) if context_test else True
    
    if tool_ok and context_ok:
        print("\n✅ This model is ready to use with Strix!")
        sys.exit(0)
    else:
        print("\n❌ This model has issues that need to be resolved.")
        sys.exit(1)


if __name__ == "__main__":
    main()