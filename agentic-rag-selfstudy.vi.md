# Tự học: Xây dựng một Agent RAG Bộ nhớ (Agentic RAG Memory Agent)

## 0. Bạn đang bắt đầu từ đâu

Bạn đã có ba thứ hoạt động được: một Orchestrator phân loại yêu cầu của người dùng và định tuyến nó, một TodoAgent biến các task thô thành một contract đã được kiểm chứng (validated), và một DailyPlannerAgent biến contract đó thành một lịch trình (kèm theo một quyết định thực sự khi ngày bị quá tải).

Tuần này bạn thêm một **agent thứ ba**: một agent truy xuất (retrieve) và suy luận trên dữ liệu đã lưu — bao gồm chính các planning log của nó, cộng với bất kỳ tài liệu nào bạn cung cấp — để trả lời câu hỏi. Đây cũng là bài kiểm tra thực sự đầu tiên cho kiến trúc bạn đã xây dựng lần trước — việc thêm một agent chỉ nên đòi hỏi một thay đổi ở **một chỗ** (orchestrator), chứ không phải viết lại những gì đã chạy được. Nếu không phải vậy, thì có gì đó trong thiết kế đã sai.

---

## 1. Mục tiêu của tuần này

1. **Hiểu giải phẫu của một hệ thống RAG** — Ingestion (nạp dữ liệu), Retrieval (truy xuất), Generation (sinh câu trả lời) — và tự mình triển khai cả ba.
2. **Xây dựng một vòng lặp truy xuất *thực sự mang tính agentic*.** Không phải kiểu "gọi một hàm tìm kiếm rồi nhét kết quả vào prompt" — mà là một agent tự quyết định *có nên* truy xuất hay không, *nguồn nào* để truy vấn, *liệu* thứ trả về đã đủ chưa, và *lặp lại* cho đến khi đủ hoặc bỏ cuộc. Sự khác biệt đó chính là nội dung cốt lõi của tuần này, nên đừng bỏ qua §2.

---

## 2. Khái niệm: ba phần của RAG, và điều gì khiến nó trở thành "Agentic"

**Ingestion** (offline, trước mọi truy vấn) — biến nguyên liệu thô thành thứ có thể tìm kiếm được: chia nhỏ văn bản thành chunk → embed mỗi chunk thành một vector → lưu vector + văn bản gốc + metadata.

**Retrieval** (mỗi truy vấn) — embed câu truy vấn bằng *cùng* embedding model đã dùng lúc ingestion, rồi tìm kiếm theo độ tương đồng (similarity search) trên các vector đã lưu để lấy top-k chunk gần nhất.

**Generation** (mỗi truy vấn) — ghép các chunk truy xuất được thành một prompt, hướng dẫn model trả lời *chỉ* dựa trên ngữ cảnh đó (cùng bài học về "grounding" như "đừng bịa ra task" ở tuần 1, áp dụng cho dữ liệu này), rồi gọi LLM.

Đó là RAG tiêu chuẩn: một lượt truy xuất, một lượt sinh, luôn luôn như vậy. Nó là một pipeline cố định — cùng ba bước chạy mỗi lần, bất kể câu hỏi là gì.

**Agentic RAG khác biệt ở đúng một điểm cụ thể: agent tự kiểm soát chính quá trình truy xuất.** Thay vì luôn truy xuất một lần rồi sinh câu trả lời, agent:

- **quyết định truy vấn nguồn/tool nào** khi có nhiều hơn một lựa chọn,
- **đánh giá xem thứ nó nhận về có thực sự đủ** để trả lời hay không,
- **lặp lại** — diễn đạt lại truy vấn và truy xuất lần nữa — nếu chưa đủ, cho tới một điểm dừng nào đó.

Bài kiểm tra để áp dụng cho chính bản dựng của bạn: **nếu agent của bạn luôn thực hiện đúng một lượt truy-xuất-rồi-sinh, thì đó là RAG tiêu chuẩn, kể cả khi có LLM tham gia.** Một phép kiểm tra ngưỡng điểm tương đồng duy nhất ("nếu score > 0.5 thì trả lời; nếu không thì nói tôi không biết") là một lớp bảo vệ tốt, nhưng nó *không* mang tính agentic — vẫn chỉ là một lượt. Bạn cần một vòng lặp có một nhánh thực sự quay lại "truy xuất lần nữa" thì mới xứng với cái tên đó.

---

## 3. Các quyết định thiết kế cho dự án này

### 3.1 Bạn làm việc với dữ liệu nào

Bạn cần *quyết định* corpus (kho ngữ liệu), chứ không chỉ giả định có sẵn một cái. Xây dựng nó từ **hai nguồn**:

- **Các bản tóm tắt planning được ghi tự động** — mỗi khi DailyPlannerAgent xây xong một lịch trình, nó ghi một đoạn văn bản ngắn vào kho dữ liệu ("Vào 2026-06-19, 6 task, hoãn 2 task đọc tài liệu ưu tiên thấp"). Đây là nguồn chính — nó tạo ra một vòng phản hồi (feedback loop) giữa hai agent, và những câu hỏi như "khi bận tôi thường ưu tiên cái gì?" trở nên trả lời được.
- **Các tài liệu được tải lên** — bất kỳ tài liệu văn bản nào bạn muốn hệ thống biết đến (ghi chú, một tài liệu tham khảo, bất cứ thứ gì). Đường đi này **không gắn với agent nào cả** — bạn (hoặc một script) nạp nó vào trực tiếp, độc lập với TodoAgent hay DailyPlannerAgent.

Gắn nhãn mỗi chunk dữ liệu với một `source_type` (`"planning_log"` hoặc `"document"`) khi bạn ghi nó. Nhãn đó chính là thứ giúp việc chọn tool/nguồn trở nên khả thi về sau — không có nó, agent chẳng có gì để lựa chọn giữa các nguồn cả.

### 3.2 Memory MCP server — các tool và nó cắm vào đâu

Toàn bộ dữ liệu này nằm sau một MCP server chuyên dụng, tách biệt với task server của bạn — cùng kiến trúc như tuần 1: nó là một process riêng, và bất cứ thứ gì cần nó đều kết nối qua một MCP client, chứ không import code của nó trực tiếp.

**Nó cắm vào đâu** — ba caller khác nhau kết nối tới nó, vì những lý do khác nhau:

```
   DailyPlannerAgent          Document ingestion script          RAGAgent
          │                            │                             │
          │ remember(...)              │ remember(...)               │ retrieve_log(query)
          │ source_type=               │ source_type=                │ retrieve_document(query)
          │  "planning_log"            │  "document"                 │
          ▼                            ▼                             ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │                  MEMORY MCP SERVER   (separate process)            │
   │      tools:  remember · retrieve_log · retrieve_document           │
   └────────────────────────────────────────────────────────────────────┘
```

**Nó cung cấp các tool nào:**

- `remember(text, source_type, metadata) -> id` — ghi một chunk dữ liệu mới vào kho. Được gọi bởi DailyPlannerAgent (với `source_type="planning_log"`) và bởi script nạp tài liệu của bạn (với `source_type="document"`).
- `retrieve_log(query, k=3) -> list[DataChunk]` — tìm kiếm **chỉ** trong dữ liệu planning-log.
- `retrieve_document(query, k=3) -> list[DataChunk]` — tìm kiếm **chỉ** trong dữ liệu document.

Việc tách truy xuất thành hai tool thay vì một là có chủ đích: việc chọn tool chỉ có ý nghĩa khi có nhiều hơn một tool để chọn. Một câu hỏi như "khi bận tôi thường ưu tiên cái gì" rõ ràng gọi đến `retrieve_log`; "tài liệu tham khảo của tôi nói gì về X" rõ ràng gọi đến `retrieve_document` — và RAGAgent phải nhìn vào câu hỏi để quyết định, đó chính là nửa "chọn tool" của tính agentic.

### 3.3 Phán xét về tính đầy đủ là một quyết định, không phải một ngưỡng

Sau khi truy xuất, agent phải quyết định: tôi có thể trả lời từ những thứ này không, hay tôi cần tìm lại (và tìm gì, ở đâu)? Hãy biến quyết định đó thành **structured LLM output**, cùng kỷ luật như các contract của tuần trước — chứ không phải một con số cắt ngưỡng. Xem `RetrievalDecision` ở §7.

### 3.4 Vòng lặp cần một điều kiện dừng cứng

Một khi truy xuất có thể lặp lại, chi phí và độ trễ không còn là một con số cố định nữa mà trở thành một *phân phối* — một số truy vấn xong trong một lượt, một số mất ba lượt. Một giới hạn cứng `MAX_ITERATIONS` là không thể thiếu; một vòng lặp không giới hạn là một rủi ro production, không chỉ là chuyện thiếu hiệu quả.

---

## 4. Những gì bạn sẽ xây (5 sản phẩm bàn giao)

Xây theo thứ tự này — mỗi cái phụ thuộc vào cái trước.

| # | Sản phẩm bàn giao | Phục vụ | Vì sao quan trọng |
|---|---|---|---|
| 1 | **Contracts**: `DataChunk`, `RetrievalDecision` | kỷ luật ingestion/retrieval | Cùng vai trò như handoff contract của tuần trước — không có gì vô kỷ luật được phép vượt qua ranh giới. |
| 2 | **Memory MCP server**: `remember`, `retrieve_log`, `retrieve_document` | ingestion + retrieval | Giấu embedding model và vector store sau các tool mà bất kỳ caller nào cũng dùng được. |
| 3 | **Hai đường ingestion**: hook trong DailyPlannerAgent + script tài liệu độc lập | ingestion | Xây corpus — một cái tự động, một cái thủ công và độc lập với agent. |
| 4 | **RAGAgent**: vòng lặp retrieve → judge → generate | generation + tính chủ động | Phần agentic. Quyết định bắt buộc nằm ở đây. |
| 5 | **Wiring Orchestrator**: nhánh `recall` mới | tích hợp | Chứng minh tính chất "chỉ thêm-một-chỗ-cho-một-agent" vẫn đúng. |

---

## 5. Công cụ

- **Gợi ý embedding model `nomic-embed-text`/`all-minilm` hoặc model khác.**
- **Tự tay viết vector store trước** — một list các vector cộng với một vòng lặp cosine-similarity bằng Python/numpy thuần. Nó chỉ khoảng chục dòng và biến "vector search" thành thứ bạn *hiểu* thay vì thứ một thư viện làm hộ bạn. (Thay bằng Chroma/FAISS sau cũng được — giữ nguyên interface `retrieve_*` để không có gì ở phía trên phải đổi.)
- **Structured output của Ollama cho bước judge** — cùng mẫu như các lượt structured-output của tuần trước: sinh JSON schema từ Pydantic model của bạn, truyền nó qua `format`, kiểm chứng response.

---

## 6. Luồng dữ liệu

**Quy trình ingestion — xây dựng kho dữ liệu:**

```
   DailyPlannerAgent                         a document you provide
   finishes a schedule                       (.txt file, etc.)
          │                                          │
          │ short text summary                       ▼
          │ (no chunking needed)              CHUNK into smaller pieces
          │                                          │
          └─────────────────┬────────────────────────┘
                             ▼
                     EMBED each piece
                  (same model used at retrieval!)
                             ▼
              remember(text, source_type, metadata)
                             ▼
                  ┌─────────────────────────┐
                  │        DATA STORE       │
                  │  vector + text +        │
                  │  source_type + metadata │
                  └─────────────────────────┘
```

**Cấp Orchestrator — một nhánh mới:**

```
USER PROMPT
    │
    ▼
ORCHESTRATOR — classify: summary | plan | add task | recall   ← new branch
    │                                              │
    ▼ (existing agents)                            ▼ (new)
TodoAgent / DailyPlannerAgent                   RAGAgent
                                                   │
                                              (see loop below)
                                                   │
                                                   ▼
                                                RESPONSE
```

**Bên trong RAGAgent — vòng lặp agentic, kèm cả hai contract:**

```
            ┌───────────────┐
            │   RETRIEVE    │◄─────────────────────────┐
            │ choose source │                          │
            │ + query       │                          │
            └──────┬────────┘                          │
                   ▼                                   │
       ┌─────────────────────────────┐                 │
       │ CONTRACT: DataChunk[]       │                 │
       │ every tool result is checked│                 │
       │ against this shape (§7.1)   │                 │
       └──────────────┬──────────────┘                 │
                      ▼                                │
                ┌───────────────┐                      │
                │     JUDGE     │                      │
                │ enough to     │                      │
                │ answer?       │                      │
                └──────┬────────┘                      │
                       ▼                               │
       ┌───────────────────────────────┐               │
       │ CONTRACT: RetrievalDecision   │               │
       │ the judge's structured output │               │
       │ — validated before branching  │               │
       │ (§7.1)                        │               │
       └───────────────┬───────────────┘               │
                       │                               │
            sufficient?│  no, iterations left ─────────┘
                       │ yes — or out of iterations
                       ▼
                ┌───────────────┐
                │   GENERATE    │
                │ grounded      │
                │ answer        │
                └───────────────┘
```

## 7. Gợi ý triển khai

### 7.1 Contracts

```python
# contracts.py
from pydantic import BaseModel
from typing import Literal

class DataChunk(BaseModel):
    text: str
    score: float
    source_type: Literal["planning_log", "document"]
    metadata: dict

class RetrievalDecision(BaseModel):
    sufficient: bool
    reasoning: str
    next_source: Literal["logs", "documents", "both"] | None = None
    next_query: str | None = None     # reformulated, only if sufficient=False
```

### 7.2 Data store (tự viết)

```python
# store.py
import numpy as np
from embeddings import embed

class DataStore:
    def __init__(self, path="data/store.json"):
        self.path = path
        self.chunks = self._load()   # list of {text, vector, source_type, metadata}

    def add(self, text: str, source_type: str, metadata: dict) -> str:
        # TODO: embed(text), append to self.chunks, persist, return an id
        ...

    def search(self, query: str, k: int, source_type: str | None = None) -> list[dict]:
        # TODO: embed(query), cosine-similarity against self.chunks
        #       (filter by source_type if given), sort, return top k
        ...
```

### 7.3 Script ingestion — tài liệu

```python
# ingest_documents.py — standalone script, NOT tied to any agent.
# Run it manually whenever you want to add a document to the data store.

def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    # TODO: split `text` into ~chunk_size-character pieces
    #       (simplest approach: split on paragraphs, group until you hit chunk_size)
    ...

def ingest(file_path: str):
    text = open(file_path).read()
    for chunk in chunk_text(text):
        # TODO: call the Memory MCP client's `remember` tool with this chunk,
        #       source_type="document", metadata={"file": file_path}
        ...
```

### 7.4 RAGAgent — vòng lặp

```python
# rag_agent.py
class RAGAgent:
    MAX_ITERATIONS = 3

    def run(self, query: str) -> str:
        collected: list[dict] = []     # accumulate — never discard between iterations
        current_query, source = query, "both"

        for i in range(self.MAX_ITERATIONS):
            # TODO: call retrieve_log / retrieve_document / both depending on `source`,
            #       extend `collected` with the results (validate each against DataChunk)
            decision = self._judge(query, collected)   # TODO: structured Ollama call
                                                          # returning RetrievalDecision
            log.info("rag_step", iter=i, query=current_query,
                     source=source, decision=decision.model_dump())

            if decision.sufficient:
                return self._generate(query, collected)   # TODO: grounded LLM answer

            current_query = decision.next_query or current_query
            source = decision.next_source or source

        # TODO: stopping condition — out of iterations.
        #       Answer from whatever was collected (with a caveat) if there's
        #       anything useful, otherwise say so honestly. Do not loop forever.
```

---

## 8. Định nghĩa "hoàn thành" (Definition of done)

- [ ] `remember`, `retrieve_log`, và `retrieve_document` đều chạy được đầu-cuối (end-to-end) qua Memory MCP server, với `embed()` được dùng chung giữa đường ghi và đường đọc.
- [ ] Mọi chunk dữ liệu đều được gắn nhãn `source_type` (`planning_log` hoặc `document`) tại thời điểm ghi.
- [ ] DailyPlannerAgent tự động ghi một mục `planning_log` sau khi xây xong lịch trình.
- [ ] Bạn có thể ingest ít nhất một tài liệu một cách độc lập với các agent (qua script độc lập) và truy xuất thành công từ nó.
- [ ] RAGAgent chọn `retrieve_log` hay `retrieve_document` (hoặc cả hai) dựa trên câu hỏi — demo một truy vấn rõ ràng trúng mỗi loại.
- [ ] Một truy vấn mà lần truy xuất đầu yếu sẽ kích hoạt một truy vấn thứ hai **được diễn đạt lại** — khác biệt rõ rệt so với lần đầu trong trace log.
- [ ] `MAX_ITERATIONS` được thực thi; một truy vấn không bao giờ có thể thỏa mãn vẫn kết thúc và trả lời một cách trung thực thay vì lặp vô hạn hoặc crash.
- [ ] Mọi kết quả tool đều được kiểm chứng theo `DataChunk`, và mọi lượt judge theo `RetrievalDecision`, trước khi vòng lặp rẽ nhánh dựa trên nó.
- [ ] Trace log của bất kỳ lần chạy đơn lẻ nào cũng cho phép bạn tái dựng lại mọi truy vấn đã phát ra, nguồn đã chọn, và phán xét đã đưa ra.
- [ ] Orchestrator chỉ cần đúng **một** nhánh mới để thêm RAGAgent — chỉ ra diff đó.

---
