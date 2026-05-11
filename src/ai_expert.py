import hashlib
import json
import os
import re
import shutil
from datetime import datetime

from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


class CareerAIExpert:
    def __init__(self, api_key, pdf_path="data/documents", chunk_size=1000, chunk_overlap=100):
        # 1. Cấu hình API Key
        os.environ["GOOGLE_API_KEY"] = api_key

        # 2. Cấu hình ingest và nơi lưu index/manifest
        self.embedding_model = "gemini-embedding-001"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = GoogleGenerativeAIEmbeddings(model=self.embedding_model)

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db_path = os.path.join(project_root, "data", "faiss_index")
        self.manifest_path = os.path.join(self.db_path, "manifest.json")
        self.per_file_indexes_dir = os.path.join(self.db_path, "per_file_indexes")
        self.manifest_version = 2

        # 3. Đảm bảo thư mục data/documents tồn tại
        if not os.path.exists(pdf_path):
            os.makedirs(pdf_path)
            print(f"Created empty document directory at: {pdf_path}. Add PDF files there.")

        # 4. Kiểm tra xem có cần ingest lại không
        os.makedirs(self.db_path, exist_ok=True)
        os.makedirs(self.per_file_indexes_dir, exist_ok=True)
        current_files = self._scan_pdf_files(pdf_path)
        if not current_files:
            raise FileNotFoundError(f"Không tìm thấy file PDF nào trong thư mục {pdf_path}")

        manifest = self._load_manifest()
        plan = self._build_update_plan(current_files, manifest)

        if plan["rebuild_all"] or plan["files_to_index"] or plan["files_to_remove"]:
            print("PDF data changed. Updating per-file FAISS indexes...")
            for reason in plan["reasons"]:
                print(f"   - {reason}")
            self._apply_index_updates(current_files, manifest, plan)
        else:
            print("PDF files unchanged. Reusing existing per-file FAISS indexes.")

        # 5. Load database sau khi đã có index
        print("Loading and merging per-file FAISS indexes...")
        self.vector_db = self._load_merged_vector_db()
        print("RAG Expert is ready.")

    def _file_signature(self, file_path):
        file_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                file_hash.update(chunk)

        stat = os.stat(file_path)
        return {
            "path": file_path,
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "sha256": file_hash.hexdigest(),
        }

    def _scan_pdf_files(self, folder_path):
        files = {}
        for file_name in sorted(os.listdir(folder_path)):
            if not file_name.lower().endswith(".pdf"):
                continue
            full_path = os.path.join(folder_path, file_name)
            files[file_name] = self._file_signature(full_path)
        return files

    def _load_manifest(self):
        if not os.path.exists(self.manifest_path):
            return None

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Cannot read existing manifest: {e}")
            return None

    def _safe_index_dir_name(self, file_name):
        stem, _ = os.path.splitext(file_name)
        safe_stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem).strip("_") or "pdf"
        short_hash = hashlib.sha1(file_name.encode("utf-8")).hexdigest()[:10]
        return f"{safe_stem}_{short_hash}"

    def _get_file_index_dir(self, file_name):
        return os.path.join(self.per_file_indexes_dir, self._safe_index_dir_name(file_name))

    def _resolve_index_dir(self, entry):
        index_dir = entry.get("index_dir", "")
        if not index_dir:
            return None
        if os.path.isabs(index_dir):
            return index_dir
        return os.path.join(self.db_path, index_dir)

    def _has_index_dir(self, index_dir):
        if not index_dir:
            return False
        return (
            os.path.exists(os.path.join(index_dir, "index.faiss"))
            and os.path.exists(os.path.join(index_dir, "index.pkl"))
        )

    def _build_update_plan(self, current_files, manifest):
        reasons = []
        plan = {
            "rebuild_all": False,
            "files_to_index": [],
            "files_to_remove": [],
            "reasons": reasons,
        }

        if manifest is None:
            plan["rebuild_all"] = True
            plan["files_to_index"] = sorted(current_files.keys())
            reasons.append("Chưa có manifest.json, cần tạo index riêng cho từng file.")
            return plan

        if manifest.get("version") != self.manifest_version:
            plan["rebuild_all"] = True
            plan["files_to_index"] = sorted(current_files.keys())
            reasons.append("Manifest đang ở phiên bản cũ, cần chuyển sang index riêng theo từng PDF.")
            return plan

        embedding_changed = manifest.get("embedding_model") != self.embedding_model
        chunking_changed = (
            manifest.get("chunk_size") != self.chunk_size
            or manifest.get("chunk_overlap") != self.chunk_overlap
        )
        if embedding_changed or chunking_changed:
            plan["rebuild_all"] = True
            plan["files_to_index"] = sorted(current_files.keys())
            if embedding_changed:
                reasons.append("Embedding model đã thay đổi.")
            if chunking_changed:
                reasons.append("Cấu hình chunk_size/chunk_overlap đã thay đổi.")
            return plan

        manifest_files = manifest.get("files", {})
        current_names = set(current_files.keys())
        manifest_names = set(manifest_files.keys())

        removed_files = sorted(manifest_names - current_names)
        if removed_files:
            plan["files_to_remove"] = removed_files
            reasons.append(
                f"Có file PDF bị xóa/không còn trong thư mục: {', '.join(removed_files)}"
            )

        files_to_index = []
        for file_name in sorted(current_names):
            current_info = current_files[file_name]
            manifest_info = manifest_files.get(file_name)

            if manifest_info is None:
                files_to_index.append(file_name)
                reasons.append(f"Có file PDF mới: {file_name}")
                continue

            index_dir = self._resolve_index_dir(manifest_info)
            if not self._has_index_dir(index_dir):
                files_to_index.append(file_name)
                reasons.append(f"Thiếu FAISS index riêng cho file: {file_name}")
                continue

            if (
                manifest_info.get("sha256") != current_info["sha256"]
                or manifest_info.get("size") != current_info["size"]
                or abs(float(manifest_info.get("mtime", 0)) - float(current_info["mtime"])) > 1e-6
                or manifest_info.get("status") != "indexed"
            ):
                files_to_index.append(file_name)
                reasons.append(f"File PDF đã thay đổi hoặc index lỗi: {file_name}")

        plan["files_to_index"] = files_to_index
        return plan

    def _save_manifest(self, manifest_data):
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2)

    def _remove_index_dir(self, index_dir):
        if index_dir and os.path.exists(index_dir):
            shutil.rmtree(index_dir, ignore_errors=True)

    def _index_single_file(self, file_name, file_info, splitter):
        file_full_path = file_info["path"]
        print(f"Updating data from file: {file_name}...")
        index_dir = self._get_file_index_dir(file_name)
        self._remove_index_dir(index_dir)
        os.makedirs(index_dir, exist_ok=True)

        loader = PyPDFLoader(file_full_path)
        file_documents = loader.load()
        file_chunks = splitter.split_documents(file_documents)

        for idx, chunk in enumerate(file_chunks):
            chunk.metadata["source"] = file_name
            chunk.metadata["file_sha256"] = file_info["sha256"]
            chunk.metadata["chunk_id"] = idx

        print(f"Creating embeddings for {len(file_chunks)} chunks from file {file_name}...")
        vector_db = FAISS.from_documents(file_chunks, self.embeddings)
        vector_db.save_local(index_dir)
        print(f"Saved per-file FAISS index for {file_name} at: {index_dir}")

        return {
            "path": file_full_path,
            "size": file_info["size"],
            "mtime": file_info["mtime"],
            "sha256": file_info["sha256"],
            "pages": len(file_documents),
            "chunks": len(file_chunks),
            "status": "indexed",
            "indexed_at": datetime.utcnow().isoformat() + "Z",
            "index_dir": os.path.relpath(index_dir, self.db_path),
        }

    def _apply_index_updates(self, current_files, manifest, plan):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        manifest_files = {} if manifest is None or plan["rebuild_all"] else dict(manifest.get("files", {}))

        if plan["rebuild_all"]:
            shutil.rmtree(self.per_file_indexes_dir, ignore_errors=True)
            os.makedirs(self.per_file_indexes_dir, exist_ok=True)
            manifest_files = {}

        for file_name in plan["files_to_remove"]:
            old_entry = manifest_files.pop(file_name, None)
            self._remove_index_dir(self._resolve_index_dir(old_entry or {}))

        for file_name in plan["files_to_index"]:
            old_entry = manifest_files.get(file_name)
            self._remove_index_dir(self._resolve_index_dir(old_entry or {}))
            file_info = current_files[file_name]

            try:
                manifest_files[file_name] = self._index_single_file(file_name, file_info, splitter)
            except Exception as e:
                print(f"Error while creating per-file index for {file_name}: {e}")
                manifest_files[file_name] = {
                    "path": file_info["path"],
                    "size": file_info["size"],
                    "mtime": file_info["mtime"],
                    "sha256": file_info["sha256"],
                    "status": "failed",
                    "error": str(e),
                    "index_dir": os.path.relpath(self._get_file_index_dir(file_name), self.db_path),
                }

        # Đồng bộ lại metadata cho các file không thay đổi nhưng vẫn còn tồn tại
        for file_name in sorted(set(current_files.keys()) - set(plan["files_to_index"])):
            if file_name in manifest_files:
                manifest_files[file_name]["path"] = current_files[file_name]["path"]
                manifest_files[file_name]["size"] = current_files[file_name]["size"]
                manifest_files[file_name]["mtime"] = current_files[file_name]["mtime"]
                manifest_files[file_name]["sha256"] = current_files[file_name]["sha256"]

        manifest_data = {
            "version": self.manifest_version,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "indexed_at": datetime.utcnow().isoformat() + "Z",
            "files": manifest_files,
        }
        self._save_manifest(manifest_data)
        print(f"Updated manifest at: {self.manifest_path}")

    def _load_merged_vector_db(self):
        manifest = self._load_manifest()
        if manifest is None:
            raise RuntimeError("Không tìm thấy manifest sau khi cập nhật FAISS.")

        merged_db = None
        indexed_files = []
        for file_name in sorted(manifest.get("files", {}).keys()):
            entry = manifest["files"][file_name]
            if entry.get("status") != "indexed":
                continue

            index_dir = self._resolve_index_dir(entry)
            if not self._has_index_dir(index_dir):
                print(f"Skipping file {file_name} because its per-file index is missing.")
                continue

            current_db = FAISS.load_local(
                index_dir,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            if merged_db is None:
                merged_db = current_db
            else:
                merged_db.merge_from(current_db)
            indexed_files.append(file_name)

        if merged_db is None:
            raise RuntimeError("Không có FAISS index hợp lệ nào để nạp.")

        print(f"Loaded and merged {len(indexed_files)} per-file FAISS indexes.")
        return merged_db

    def chat_with_admission(self, question):
        """Hỏi đáp về quy chế tuyển sinh dựa trên PDF (RAG)"""
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
            # Sử dụng as_retriever để tìm kiếm thông tin liên quan nhất
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="stuff", 
                retriever=self.vector_db.as_retriever(search_kwargs={"k": 5})
            )
            response = qa_chain.invoke({"query": question})
            return response["result"]
        except Exception as e:
            return f"⚠️ Có lỗi xảy ra khi xử lý câu hỏi: {str(e)}"

# --- CÁCH CHẠY THỬ (SỬ DỤNG TRONG FILE TEST_API.PY) ---
if __name__ == "__main__":
    # Thay API Key của bạn vào đây
    MY_KEY = "AIzaSyCqo5MWq-zM3BBlqTYbUdP5oFnEzrHWfN8" 
    expert = CareerAIExpert(api_key=MY_KEY)
    
    # Test một câu hỏi bất kỳ
    ans = expert.chat_with_admission("Năm 2025 trường dự kiến tuyển bao nhiêu chỉ tiêu?")
    print(f"\nAI response: {ans}")
