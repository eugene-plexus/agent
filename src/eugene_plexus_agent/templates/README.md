# DeepSeek V4 chat template

`deepseek-v4.jinja` is the unmodified llama.cpp **b11719** template:

- Source: <https://github.com/ggml-org/llama.cpp/blob/b11719/models/templates/deepseek-ai-DeepSeek-V4.jinja>
- Upstream Git blob: `c7819149ae6e73671331c02f03b603cd26d22ef5`
- SHA-256: `7bd4fbdf0c06cd7398ddb6913de6e4a5839b9d06569e74e5b2ec246b375ca9c9`
- License: MIT; the upstream notice is in `LICENSE.llama.cpp`.

Reviewed against DeepSeek's V4 format at `935817c66f5d6e64838f419692545f06787e485f`:
<https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/deepseek-recipe-encoding/src/v4/dsv4.rs>.
The review covers BOS/system/user/assistant boundaries, non-thinking and
thinking generation prompts, DSML string versus JSON arguments, and tool
results. The llama.cpp template also retains its upstream reasoning-effort
and response-format options. It is an engine Jinja template, not a copy of
DeepSeek's request-conversion and token-encoding library.

Only header-confirmed DeepSeek V4 uses this fallback. V4.1 and Flash-0731
have different formats and do not select it. Their launch is still allowed.
An operator's template always wins; an embedded template wins next.

llama.cpp b11719 detects the template's `dsml_token` and `tool_calls` and
uses `common_chat_params_init_deepseek_v3_2` in `common/parsers/deepseek.cpp`
for native incremental and complete tool parsing. vLLM v0.29.0 has
`vllm/parser/deepseek_v4.py`. No missing V4 parser was found at these
revisions, so Eugene adds no second parser. The published vLLM format
example is recorded in `tests/fixtures/deepseek-v4.json`; offline tests
render its argument types and tool-result turns. Native execution and tool
round trips still require the hardware acceptance run.
