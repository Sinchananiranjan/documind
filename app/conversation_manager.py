import os
import json
import uuid
import time
import re
from typing import Dict, List, Any, Optional

CONVERSATIONS_FILE = "./data/conversations.json"

class ConversationManager:
    """Manages local storage of conversation histories and conversation-scoped document metadata with lazy persistence."""

    def __init__(self, storage_file: str = CONVERSATIONS_FILE):
        self.storage_file = storage_file
        os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
        self.drafts: Dict[str, Any] = {}
        self.conversations: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Filter out empty saved conversations if any existed previously
                    cleaned = {k: v for k, v in data.items() if len(v.get("messages", [])) > 0 or len(v.get("documents", [])) > 0}
                    return cleaned
            except Exception:
                return {}
        return {}

    def _save(self):
        # Only save non-empty conversations to disk
        persisted = {k: v for k, v in self.conversations.items() if len(v.get("messages", [])) > 0 or len(v.get("documents", [])) > 0}
        with open(self.storage_file, "w", encoding="utf-8") as f:
            json.dump(persisted, f, indent=2)

    def create_conversation(self, mode: str = "auto", save_immediately: bool = False) -> str:
        """
        Creates a new conversation instance.
        By default (save_immediately=False), creates an in-memory draft that is NOT persisted
        to disk until the user sends a message or attaches a document.
        """
        conv_id = str(uuid.uuid4())
        conv_obj = {
            "id": conv_id,
            "title": "New Conversation",
            "timestamp": time.time(),
            "mode": mode,
            "messages": [],
            "active_docs": [],
            "documents": []
        }
        if save_immediately:
            self.conversations[conv_id] = conv_obj
            self._save()
        else:
            self.drafts[conv_id] = conv_obj
        return conv_id

    def get_conversation(self, conv_id: str) -> Optional[Dict[str, Any]]:
        if conv_id in self.conversations:
            return self.conversations[conv_id]
        if conv_id in self.drafts:
            return self.drafts[conv_id]
        return None

    def _ensure_persisted(self, conv_id: str):
        """Promotes an in-memory draft to persisted conversations storage."""
        if conv_id in self.drafts and conv_id not in self.conversations:
            self.conversations[conv_id] = self.drafts.pop(conv_id)
        elif conv_id not in self.conversations:
            # Fallback object
            self.conversations[conv_id] = {
                "id": conv_id,
                "title": "New Conversation",
                "timestamp": time.time(),
                "mode": "auto",
                "messages": [],
                "active_docs": [],
                "documents": []
            }

    def update_conversation(self, conv_id: str, data: Dict[str, Any]):
        self._ensure_persisted(conv_id)
        if conv_id in self.conversations:
            self.conversations[conv_id].update(data)
            self._save()

    def add_message(self, conv_id: str, message: Dict[str, Any]):
        self._ensure_persisted(conv_id)
        conv = self.conversations[conv_id]
        # Auto-generate title from first user message if title is default
        if len(conv["messages"]) == 0 and message.get("role") == "user":
            content = message.get("content", "")
            title = content[:30] + "..." if len(content) > 30 else content
            conv["title"] = title
        conv["messages"].append(message)
        self._save()

    def add_active_doc(self, conv_id: str, doc_id: str, filename: str = "", page_count: Optional[int] = None):
        """Associates a document with a conversation and tracks upload order & metadata."""
        self._ensure_persisted(conv_id)
        conv = self.conversations[conv_id]
        if "active_docs" not in conv:
            conv["active_docs"] = []
        if "documents" not in conv:
            conv["documents"] = []

        if doc_id not in conv["active_docs"]:
            conv["active_docs"].append(doc_id)
            upload_order = len(conv["documents"]) + 1
            doc_meta = {
                "doc_id": doc_id,
                "filename": filename or doc_id,
                "upload_order": upload_order,
                "timestamp": time.time(),
                "page_count": page_count
            }
            conv["documents"].append(doc_meta)
            self._save()
        elif filename:
            # Update existing filename if previously missing
            for doc in conv["documents"]:
                if doc.get("doc_id") == doc_id and not doc.get("filename"):
                    doc["filename"] = filename
                    self._save()
                    break

    def get_documents(self, conv_id: str) -> List[Dict[str, Any]]:
        """Returns ordered document metadata list for a conversation."""
        conv = self.get_conversation(conv_id)
        if conv:
            return conv.get("documents", [])
        return []

    def resolve_target_docs(self, conv_id: str, question: str) -> List[str]:
        """
        Dynamically resolves natural language references (e.g., 'first PDF', 'second PDF',
        'previous document', 'filename.pdf', 'both PDFs', 'the PDF I uploaded earlier')
        to specific doc_ids within the conversation.
        If no specific document reference is matched, returns all active_docs for the conversation.
        """
        conv = self.get_conversation(conv_id)
        if not conv:
            return []

        active_docs = conv.get("active_docs", [])
        documents = conv.get("documents", [])

        if not active_docs or not documents:
            return active_docs

        q_lower = question.lower()

        # 0. Check for 'both', 'all', 'every' multi-document references
        if (re.search(r'\b(both|all|every)\b', q_lower) and re.search(r'\b(pdfs?|docs?|documents?|files?)\b', q_lower)) or "both documents" in q_lower or "both pdfs" in q_lower or "all documents" in q_lower:
            return active_docs

        # 1. Filename matching (exact or substring)
        matched_by_filename = []
        for doc in documents:
            fname = doc.get("filename", "").lower()
            if fname:
                name_without_ext = os.path.splitext(fname)[0]
                if fname in q_lower or (len(name_without_ext) >= 3 and name_without_ext in q_lower):
                    matched_by_filename.append(doc["doc_id"])
        if matched_by_filename:
            return matched_by_filename

        # 2. Ordinal mapping (first/1st, second/2nd, third/3rd, etc.)
        ordinals = {
            "first": 1, "1st": 1,
            "second": 2, "2nd": 2,
            "third": 3, "3rd": 3,
            "fourth": 4, "4th": 4,
            "fifth": 5, "5th": 5
        }

        matched_ordinals = set()
        for word, order in ordinals.items():
            pattern = r'\b' + word + r'\b'
            if re.search(pattern, q_lower):
                matched_ordinals.add(order)

        if matched_ordinals:
            target_ids = []
            for doc in documents:
                if doc.get("upload_order") in matched_ordinals:
                    target_ids.append(doc["doc_id"])
            if target_ids:
                return target_ids

        # 3. Relative / temporal references ('uploaded earlier', 'earlier PDF', 'first uploaded')
        if re.search(r'\b(earlier|first\s+uploaded|uploaded\s+earlier)\b', q_lower):
            if documents:
                return [documents[0]["doc_id"]]

        # 4. Recent / previous references ('previous document', 'last PDF', 'latest file')
        if re.search(r'\b(previous|last|latest|recent)\s+(?:pdf|doc|document|file)\b', q_lower):
            if documents:
                return [documents[-1]["doc_id"]]

        # Default: return all active docs in conversation
        return active_docs

    def delete_conversation(self, conv_id: str) -> List[str]:
        """Deletes conversation and returns active_docs list for vector cleanup."""
        deleted_docs = []
        if conv_id in self.conversations:
            deleted_docs = list(self.conversations[conv_id].get("active_docs", []))
            del self.conversations[conv_id]
            self._save()
        if conv_id in self.drafts:
            deleted_docs.extend(self.drafts[conv_id].get("active_docs", []))
            del self.drafts[conv_id]
        return deleted_docs

    def list_conversations(self) -> List[Dict[str, Any]]:
        """Returns list of persisted non-empty conversations sorted by timestamp."""
        convs = [c for c in self.conversations.values() if len(c.get("messages", [])) > 0 or len(c.get("documents", [])) > 0]
        convs.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return convs
