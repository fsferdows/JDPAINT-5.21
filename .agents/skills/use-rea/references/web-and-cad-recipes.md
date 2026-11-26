# REA Web, Bundle & CAD Reconstruction Recipes

Exact JSON payloads verified against `/Users/user/.gemini/antigravity/mcp/rea/*.json`.

## 1. Launch Controlled Headless Chrome Inspection

```json
{
  "browser": {
    "mode": "launch",
    "executable_path": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "headless": true
  },
  "start_url": {
    "url": "https://example.com/workspace",
    "query": [
      { "name": "tab", "value": { "source": "literal", "value": "image-to-3d" } }
    ]
  },
  "actions": [
    { "step_id": "settle", "action": "wait_for_timeout", "duration_ms": 4000 }
  ],
  "capture": {
    "at_end": ["url", "accessibility", "screenshot"],
    "events": ["network"],
    "network": { "header_values": false, "request_body": false, "response_body": false }
  }
}
```

## 2. Attach Passively to User's Chrome Session (CDP Port 9222)

Call `list_browser_targets`:

```json
{
  "cdp_endpoint": "http://127.0.0.1:9222"
}
```

Then pass the returned `target_id` to `inspect_web_page` or `analyze_web_bundle`:

```json
{
  "cdp_endpoint": "http://127.0.0.1:9222",
  "target_id": "<TARGET_ID>",
  "include_script_sources": true,
  "fetch_source_maps": false,
  "observation_ms": 1500
}
```

Note: When discovering API origins in SPAs, pass both the page origin and discovered API origins inside `allowed_origins` on subsequent captures (since supplying `allowed_origins` replaces default same-origin scope).

## 3. Page Retained Evidence Without Re-Running Analysis

Use `inspect_analysis_view` with `normalized_result.parent_evidence_id`:

```json
{
  "source": {
    "kind": "retained-evidence",
    "evidence_id": "ev_<64-hex-id>"
  },
  "view": {
    "kind": "page",
    "collection": "modules",
    "offset": 0,
    "limit": 32
  }
}
```

Other high-value views:
- `{"kind": "summary"}`
- `{"kind": "api-surface", "offset": 0, "limit": 50}`
- `{"kind": "javascript", "collection": "modules", "projection": "references", "where": {"kind": "javascript-module", "path": "main.js"}, "offset": 0, "limit": 64}`

## 4. Offline JavaScript / Bundle CLI Fallback

When inspecting a local extracted app or exported script folder outside MCP:

```bash
/Users/user/.hermes/node/bin/rea analyze-javascript-application "/absolute/path/to/app" --json
```

When full deep documentation is needed across ELF, Mach-O, Android APK, or firmware targets, read `/Users/user/.gemini/config/skills/reverse-engineer-anything/SKILL.md`.
