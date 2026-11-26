# AI Agent CAD Tools & Developer Skills Guide (JDPaint 5.21 Integration)

Verified live against GitHub, PyPI, and Apple Silicon (`arm64` Python 3.12) in `/Users/user/Desktop/JDPaint 5.21`.

---

## ১. JDPaint 5.21-এর সাথে OpenCascade (`build123d` + `agentcad`) ইন্টিগ্রেশন

JDPaint-এর ব্রাউজার-ভিত্তিক **2D Vector + Hybrid AI Depth-Anything V2 + Meshy 3D** সিস্টেমের পাশাপাশি এখন একটি **স্থানীয় OpenCascade B-Rep CAD কার্নেল ব্রিজ** (`cad-kernel/`) যুক্ত করা হয়েছে:

- **শুরু করার উপায়:** ডাবল-ক্লিক করুন [Start JDPaint.command](file:///Users/user/Desktop/JDPaint%205.21/Start%20JDPaint.command) অথবা আলাদাভাবে চালান:
  ```bash
  "/Users/user/Desktop/JDPaint 5.21/cad-kernel/.venv/bin/python" "/Users/user/Desktop/JDPaint 5.21/cad-kernel/cad_bridge_server.py"
  ```
- **স্ট্যাটিক ওয়েব (Vercel) মোড:** অনলাইনে Vercel-এ ব্রাউজার একা চালু থাকলে UI কোনো এরর ছাড়াই **`Offline (Browser Mode)`** দেখাবে এবং ব্রাউজার Neural AI + SDF ইঞ্জিন কাজ চালিয়ে যাবে।
- **নতুন ফিচার (`3D Asset` ট্যাব -> `Parametric CAD Kernel`):**
  1. **`2D Vectors -> B-Rep` (`#btn-brep-vectors`):** JDPaint ক্যানভাসের 2D ভেক্টর শেপ (`rect`, `circle`, `polyline`, `curve` + ভেতরের `holes`) থেকে আসল OpenCascade B-Rep সলিড তৈরি করে।
  2. **`Parametric Part` (`#btn-brep-preset`):** তাৎক্ষণিক প্যারামেট্রিক ইন্ডাস্ট্রিয়াল পার্ট (`CNC Gusseted L-Bracket`, `Rotary Bearing Flange`, `PCB Controller Enclosure`, `Raised-Panel Door B-Rep`)।
  3. **`Export Industrial .STEP (OpenCascade)` (`#btn-asset-export-step` / `File -> Export OpenCascade B-Rep (.STEP)`):** SolidWorks / Fusion 360 / FreeCAD / Rhino / মাস্টারক্যামে ব্যবহারযোগ্য আসল NURBS ISO 10303-21 `.STEP` ফাইল ডাউনলোড করুন।

---

## ২. আপনার লিস্টের CAD টুলগুলোর তুলনামূলক বিশ্লেষণ

| টুল | ভিত্তি / কার্নেল | শক্তি ও সেরা ব্যবহার | JDPaint ও Mac-এর সাথে সামঞ্জস্য |
| :--- | :--- | :--- | :--- |
| **`jdilla1277/agentcad`** (`0.8.1`) | `build123d` (`>=0.10,<0.11`) + `cadquery-ocp` (`7.8.x`) | CLI + MCP (`python -m agentcad.mcp`), `.STEP` ও `.STL/.GLB/.OBJ` এক্সপোর্ট, `BRepCheck` ভ্যালিডেশন, `measure`, `diff`, ও স্কিল জেনারেটর। | **ইনস্টল ও ভেরিফাইড:** `cad-kernel/.venv`-এ ইনস্টল করা, `~/.gemini/config/mcp_config.json`-এ যুক্ত। Python `>=3.10,<3.13` প্রয়োজন। |
| **`pzfreo/build123d-mcp`** | `build123d` + `mcp >=2,<3` | স্ট্যান্ডঅ্যালোন MCP সার্ভার ও পরিচ্ছন্ন `b123d-*` ড্রয়িং/মডেলিং/রিপেয়ার স্কিল। | **৪টি Markdown স্কিল ইনস্টলড**; লক্ষ্য করুন এর `mcp>=2` পিন `agentcad`-এর `mcp<2`-এর সাথে এক venv-এ মিশাবেন না। |
| **`cyberchitta/cad-khana`** | `build123d` + ডায়াগনস্টিক্স ফার্স্ট JSON | মেকানিজম ইন্টারফেরেন্স/ক্লিয়ারেন্স অ্যাসারশন (`khana check`) ও 3D প্রিন্টযোগ্যতা যাচাই। | **নিরাপদ SKILL.md ও রেফারেন্স ইনস্টলড।** মেকানিক্যাল জয়েন্ট ও হিঞ্জ চেকের জন্য সেরা। |
| **`itsJeremyMax/solidifai`** | `build123d` + VTK + ডুয়ালিট্স MCP | আলাদা ওয়ার্কস্পেস, স্লাইডার ও প্যারামেট্রিক রিয়েলটাইম রিভিউ। | সম্পূর্ণ ওয়ার্কস্পেস অ্যাপ হিসেবে ভালো, তবে JDPaint-এর ভেতরে `agentcad` বেশি লাইটওয়েট ও কন্ট্রোলেবল। |
| **`JonFermin/cadsmith`** | `build123d` + Three.js প্রিভিউ | `/make-model`, `/edit-model`, `/export-model` স্ল্যাশ ওয়ার্কফ্লো প্যাটার্ন। | গিটহাবে যাচাইকৃত; আমাদের প্যারামেট্রিক প্রিসেট + `.GLB` প্রিভিউ ডিজাইনে রেফারেন্স হিসেবে উপযোগী। |
| **`fingerskier/build123d-claude-plugin`** | Claude Code MCP | ৭টি টুলের প্লাগইন। | সাধারণ OpenCascade MCP হিসেবে কার্যকর; Antigravity/Gemini-তে `agentcad` যথেষ্ট। |

**সিদ্ধান্ত (CNC ও 3D অ্যাসেট ওয়ার্কফ্লোর জন্য):**
- অর্গানিক রিলিফ, ছবি থেকে দরজা/খোদাই (`Auto-Trace 2D`, `Khoj24`, `Depth-Anything V2`) ও দ্রুত `.STL`/`.GLB` তৈরিতে **JDPaint ব্রাউজার ইঞ্জিন** ব্যবহার করুন।
- নিখুঁত মিলিমিটার মাপের মেকানিক্যাল ব্র্যাকেট, এনক্লোজার, ফ্ল্যাঞ্জ ও `.STEP` ফাইলের জন্য **OpenCascade (`build123d` + `agentcad`) ব্রিজ** ব্যবহার করুন।

---

## ৩. নিরাপদভাবে যাচাই ও ইনস্টল করা স্কিল এবং MCP সার্ভার

নিরাপত্তার জন্য প্রতিটি স্কিলের `SKILL.md` ও স্ক্রিপ্ট আগে পড়ে নিয়ে তারপর কেবল বিশ্বস্ত ও প্রয়োজনীয় স্কিলগুলোই ইনস্টল করা হয়েছে (Claude-নির্দিষ্ট রিস্কি অটো-হুক স্ক্রিপ্ট বা আনভেরিফাইড পারসোনা বট এড়িয়ে যাওয়া হয়েছে):

- **Python OpenCascade পরিবেশ (`cad-kernel/.venv`, Python 3.12):**
  `agentcad==0.8.1`, `build123d==0.10.0`, `cadquery-ocp==7.8.1.1.post1`, `mcp==1.30.0`, `trimesh==5.2.2`, `shapely==2.2.0`, `networkx==3.7`, `vtk==9.3.1`, `uvicorn==0.54.0`
- **MCP সার্ভার কনফিগ ([mcp_config.json](file:///Users/user/.gemini/config/mcp_config.json)):**
  `"agentcad": { "command": ".../cad-kernel/.venv/bin/python", "args": ["-m", "agentcad.mcp"] }` যুক্ত করা হয়েছে।
- **ইনস্টলকৃত CAD ও প্ল্যানিং স্কিল (`.agents/skills/` ও `~/.gemini/config/skills/`):**
  - `agentcad` — `agentcad skill show` থেকে জেনারেটেড `SKILL.md`
  - `b123d-modeling`, `b123d-drawing`, `b123d-edit`, `b123d-repair` (`pzfreo/build123d-mcp` থেকে Pure Markdown)
  - `cad-khana` (`cyberchitta/cad-khana` থেকে Pure Markdown + references)
  - `planning-with-files` (`~/.gemini/config/skills/planning-with-files`, `2.43.0` Gemini-অ্যাডাপ্টেড ভ্যারিয়েন্ট)

> **নিরাপত্তা পরামর্শ:** থার্ড-পার্টি হুক-নির্ভর স্কিল (`UserPromptSubmit` / `PreToolUse` শেল হুক) বা বড় স্কিল-প্যাক (`agentic-awesome-skills`-এর ২,৮০০+ স্কিল) পুরোটা ইনস্টল করলে অপ্রয়োজনীয় কমান্ড চলতে পারে এবং কন্টেক্সট উইন্ডো পূর্ণ হয়ে যেতে পারে। সবসময় প্রয়োজনীয় ২–৫টি ফিল্টারড স্কিল পিন করা কমিট থেকে ব্যবহার করুন।
