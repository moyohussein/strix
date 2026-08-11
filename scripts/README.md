# Strix Local Model Testing Scripts

This directory contains utility scripts for working with local models in Strix.

## Test Local Model Tool Call Capability

The `test_local_model_tool_call.py` script helps you verify if your local model and inference server are properly configured to work with Strix.

### Why This Matters

Strix is entirely tool-driven: every working turn must be a **native** function/tool call with structured `tool_calls` field. If your inference server returns tool calls as plain text instead, Strix won't be able to execute them.

### Usage

```bash
# Test an Ollama model
python scripts/test_local_model_tool_call.py --model "qwen3:70b" --api-base "http://localhost:11434"

# Test a custom OpenAI-compatible server
python scripts/test_local_model_tool_call.py --model "local-model" --api-base "http://localhost:8000/v1"

# Test with API key (if required)
python scripts/test_local_model_tool_call.py --model "my-model" --api-base "http://localhost:8000/v1" --api-key "your-key"

# Quick summary only (skip detailed test output)
python scripts/test_local_model_tool_call.py --model "qwen3:70b" --api-base "http://localhost:11434" --quiet

# Skip context window test
python scripts/test_local_model_tool_call.py --model "qwen3:70b" --api-base "http://localhost:11434" --no-context-test
```

### What It Tests

1. **Structured Tool Call Support**: Verifies the model returns `tool_calls` field instead of tool calls as text
2. **Context Window Capacity**: Tests if the model can handle large contexts (≥16k tokens)
3. **Response Format Compatibility**: Ensures the response format matches Strix's requirements

### Expected Output

- ✅ **PASSED**: Model is compatible with Strix
- ❌ **FAILED**: Model has issues that need to be resolved

### Return Codes

- `0`: Model is ready to use with Strix
- `1`: Model has issues that need to be resolved

### Example Output

```
Testing model: qwen3:70b
API base: http://localhost:11434

============================================================
Structured Tool Call Test
============================================================
✅ PASSED
✅ Model returns structured tool calls
Tool call format: structured

============================================================
Context Window Test
============================================================
✅ PASSED
✅ Context window test passed: Model accepted 5234 token context successfully

============================================================
MODEL CAPABILITY SUMMARY
============================================================
Model: qwen3:70b
API Base: http://localhost:11434

✅ Tool Call Capability: PASSED
   Model can return structured tool_calls field

✅ Context Window: PASSED
   Model can handle large contexts (≥16k tokens)

🎉 OVERALL: MODEL IS COMPATIBLE WITH STRIX
   This model should work well with Strix for pentesting tasks.

✅ This model is ready to use with Strix!
```

### Troubleshooting

If your model fails the test, check:

1. **Tool calls as text**: Your inference server needs to parse tool tokens into structured `tool_calls`
2. **Context window too small**: Increase to 16k-32k tokens
3. **Malformed tool calls**: Lower temperature to 0.2-0.6

See the [Local Models documentation](https://docs.strix.ai/llm-providers/local) for server-specific configuration.

## Supported Inference Servers

- **Ollama**: `export OLLAMA_NUM_CTX=32768` (required)
- **llama.cpp**: Run with `--jinja --ctx-size 32768` (required)
- **vLLM**: Start with `--enable-auto-tool-choice --tool-call-parser qwen3_xml`
- **LM Studio**: Set context window to ≥16k in settings

## Requirements

- Python 3.7+
- `requests` library: `pip install requests`