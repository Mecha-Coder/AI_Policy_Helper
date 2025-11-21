<div align="center">

# 🤖 AI Policy & Product Helper  

</div>


### **⚙️ How to Setup**

**1. Clone repo:**

```bash
git clone https://github.com/Mecha-Coder/AI_Policy_Helper.git
cd AI_Policy_Helper
```

**2. Start container:**

- `make online` online mode - (Uses qwen2.5:3b LLM and nomic-embed) 
- `make offline` offline mode - (Falls back to Stud MML and LocalEmbedder)

**3. Stop container:**

`Crtl + C`


**4. Clear everything:**

`make clean`

**Note:** Always run `make clean` before switching modes to avoid conflicts.

---

### **🏛️ Architecture**

![Architecture](https://github.com/Mecha-Coder/AI_Policy_Helper/blob/main/documentation/architecture.png)

---

### **⚖️ Trade-offs**

<details>
<summary><strong>1. LLM: Ollama vs OpenAI</strong></summary>

**Decision:** Used Ollama 

**Why:** OpenAI API key issues; pivoted to fully local solution  
- ✅ **Pro:** No API costs, complete privacy, runs offline  
- ❌ **Con:** Longer setup, limited to lightweight models due to hardware constraints  
</details>


<details>
<summary><strong>2. Embedding Model: nomic-embed-text vs LocalEmbedder</strong></summary>

**Decision:** Switched from LocalEmbedder to `nomic-embed-text` via Ollama  

**Why:** Needed higher-quality semantic search  
- ✅ **Pro:** Relevance scores improved from ~0.05 → ~0.6, better retrieval accuracy  
- ❌ **Con:** Slower ingestion, larger embeddings (768-dim vs 382-dim)  
- **Note:** Vector store must match embedding dimensions (382 for offline, 768 for online mode)  
</details>


<details>
<summary><strong>3. Retrieval Strategy</strong></summary>

**Decisions:**  
- Set relevance score threshold to > 0.4 to filter low-quality chunks  
- Reduced top-k from 4 → 3 results  

**Why:** Prioritize context quality over quantity  
- ✅ **Pro:** Less noise for LLM, clearer citations  
- ❌ **Con:** May miss relevant context in edge cases  
- **Rationale:** Better to provide 3 relevant chunks than dilute with marginal matches  
</details>


<details>
<summary><strong>4. Prompt Engineering</strong></summary>

**Decision:** Added explicit instructions for concise, grounded responses

**Implementation:**
```python
prompt = "You are an agent who understands the company's products and policies. "
prompt += "Provide a direct answer in 1-2 sentences without additional text. "
prompt += "Do not fabricate information. If no answer exists in sources, state that explicitly."
```

**Why:** Prevent verbose/hallucinated responses
- ✅ **Pro:** Clean, actionable answers; reduces hallucination
- ❌ **Con:** Slightly longer context window
- **Result:** Consistently concise responses aligned with requirements

</details>


<details>
<summary><strong>5. LLM Model Selection: qwen2.5:3b vs llama3.2:3b</strong></summary>

**Decision:** Switched from `llama3.2:3b` to `qwen2.5:3b-instruct-q4_K_M`

**Why:** Llama3.2 gave inconsistent answers and verbose responses
- ✅ **Pro:** Qwen provides accurate, consistent answers to identical queries
- ❌ **Con:** ~20-30% slower inference time
- **Rationale:** Consistency and accuracy > speed for a QA assistant
</details>


<details>
<summary><strong>6. Chunking Parameters</strong></summary>

**Decision:** Reduced `chunk_size` from 700 → 40 and `chunk_overlap` from 80 → 5

**Why:** Original values too large for short policy documents in `/data`
- ✅ **Pro:** Better granularity for small documents, more precise citations
- ❌ **Con:** More chunks to process, potential context fragmentation
</details>

<details>
<summary><strong>7. UI Enhancements</strong></summary>

**Changes:**
- Styled with Tailwind CSS
- Separated admin panel from chat interface
- Auto-scroll chatbox on new messages
- Health status polling (`/api/health` every 5s)

**Trade-off:**
- ✅ **Pro:** Clean UX, real-time system status visibility
- ❌ **Con:** Polling inefficient (5s HTTP requests vs WebSocket)

</details>

<details>
<summary><strong>8. Citation Deduplication</strong></summary>

**Decision:** Remove duplicate document titles in response

**Why:** Avoid cluttering chat with repeated source names
- ✅ **Pro:** Cleaner UI, easier to scan sources
- ❌ **Con:** May obscure that multiple chunks came from same doc
</details>

---

### **🚀 What features to ship next**



<details>
<summary><strong>1.Upgrade Model Quality</strong></summary>

Improve local hardware to support larger models, or integrate production-grade APIs such as OpenAI or Anthropic for higher-quality responses.
</details>
<details>
<summary><strong>2.Real-Time Health Updates</strong></summary>

Implement WebSocket-based push notifications for system health and metrics instead of relying on periodic polling.
</details>
<details>
<summary><strong>3.Automated Data Sync</strong></summary>

Connect to the company data server and ingest updated documents nightly to keep the vector database current.
</details>
<details>
<summary><strong>4.Sensitive Data Protection</strong></summary>

Automatically mask or redact PII and sensitive company data during generation to ensure compliance and privacy.
</details>

<details>
<summary><strong>5.Enhanced Chat Experience</strong></summary>

Allow file uploads directly through the chat interface to query documents on the fly.
</details>