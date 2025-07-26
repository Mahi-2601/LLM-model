# First import os and sys at the very top
import os
import sys

# Now do the protobuf fix
if 'google' in sys.modules:
    del sys.modules['google']
    
# Add correct protobuf path
protobuf_path = os.path.join(os.path.dirname(os.__file__), 'Lib', 'site-packages', 'google', 'protobuf')
if protobuf_path not in sys.path:
    sys.path.append(protobuf_path)

# Now import other modules
import streamlit as st
import httpx
try:
    from groq import Groq
except ImportError:
    from groq.groq import Groq
import zipfile
import tempfile
import shutil
import git
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import sqlite3
import hashlib
import pickle
import logging
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Document processing imports (replacing textract)
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.warning("PyPDF2 not available. PDF processing disabled.")

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    st.warning("python-docx not available. DOCX processing disabled.")

try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    st.warning("openpyxl not available. Excel processing disabled.")

# Load environment variables
load_dotenv()

# Set page config FIRST - before any other Streamlit commands
st.set_page_config(
    page_title="LLM Code Assistant", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Groq client - FIXED VERSION with error handling
try:
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
        http_client=httpx.Client()
    )
except Exception as e:
    st.error(f"Failed to initialize Groq client: {e}")
    st.info("Please ensure GROQ_API_KEY is set in your environment variables")
    client = None

# Initialize SentenceTransformer model
@st.cache_resource
def load_embedding_model():
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        return None

embedding_model = load_embedding_model()

# Supported models and languages
MODELS = {
    "Llama3-8b": "llama3-8b-8192",
    "Llama3-70b": "llama3-70b-8192",
    "Mixtral-8x7b": "mixtral-8x7b-32768",
    "Gemma-7b": "gemma-7b-it",
    "Llama2-70b": "llama2-70b-4096"
}

LANGUAGES = [
    "Python", "JavaScript", "Java", "C++", "TypeScript",
    "Go", "Rust", "Swift", "Kotlin", "PHP", "Ruby",
    "HTML/CSS", "SQL", "R", "Shell", "PowerShell"
]

DOMAINS = [
    "Web Development", "Backend", "Mobile Apps",
    "Data Science", "AI/ML", "DevOps",
    "Cloud Services", "Game Development", "Scripting"
]

# Enhanced supported file extensions including documents
SUPPORTED_CODE_EXTENSIONS = [
    '.py', '.js', '.java', '.cpp', '.c', '.go', '.rs', '.php', '.rb', 
    '.swift', '.kt', '.ts', '.sh', '.ps1', '.sql', '.html', '.css', 
    '.r', '.m', '.jsx', '.tsx', '.vue', '.svelte', '.scala', '.cs', 
    '.vb', '.pl', '.lua', '.dart', '.elm', '.clj', '.hs', '.ml', 
    '.f90', '.pas', '.asm', '.bat', '.yml', '.yaml', '.json', '.xml',
    '.md', '.txt', '.cfg', '.ini', '.toml'
]

SUPPORTED_DOC_EXTENSIONS = ['.pdf', '.docx', '.doc', '.xlsx', '.xls', '.pptx']
SUPPORTED_EXTENSIONS = SUPPORTED_CODE_EXTENSIONS + SUPPORTED_DOC_EXTENSIONS

# Initialize session state with database integration
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model" not in st.session_state:
    st.session_state.model = MODELS["Llama3-70b"]
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 200000
if "db_path" not in st.session_state:
    st.session_state.db_path = "codebase.db"
if "chunk_cache" not in st.session_state:
    st.session_state.chunk_cache = {}
if "max_cache_size" not in st.session_state:
    st.session_state.max_cache_size = 1000

# --------------------------
# Large Codebase Manager Class
# --------------------------
class LargeCodebaseManager:
    """Handles large codebases with database storage and lazy loading"""
    
    def __init__(self, db_path="codebase.db"):
        self.db_path = db_path
        self.init_database()
        self.chunk_cache = {}
        self.max_cache_size = 1000
        
    def init_database(self):
        """Initialize SQLite database for storing file contents and embeddings"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Files table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE,
                content TEXT,
                size INTEGER,
                extension TEXT,
                hash TEXT,
                modified_time REAL
            )
        ''')
        
        # Chunks table for embeddings
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                file_id INTEGER,
                chunk_idx INTEGER,
                content TEXT,
                embedding BLOB,
                FOREIGN KEY(file_id) REFERENCES files(id)
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_extension ON files(extension)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_size ON files(size)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id)')
        
        conn.commit()
        conn.close()
    
    def scan_directory_incremental(self, path, progress_callback=None):
        """Incrementally scan directory, only processing changed files"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        processed_count = 0
        skipped_count = 0
        total_files = sum(1 for p in Path(path).rglob('*') 
                         if p.is_file() and not self.should_ignore(p) 
                         and p.suffix.lower() in SUPPORTED_EXTENSIONS)
        
        if total_files == 0:
            conn.close()
            return 0, 0
        
        for file_path in Path(path).rglob('*'):
            if not file_path.is_file() or self.should_ignore(file_path):
                continue
                
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            
            # Check if file needs processing
            try:
                file_stat = file_path.stat()
                file_hash = self.get_file_hash(file_path)
                
                cursor.execute(
                    "SELECT id, hash, modified_time FROM files WHERE path = ?",
                    (str(file_path),)
                )
                existing = cursor.fetchone()
                
                # Skip if file hasn't changed
                if existing and existing[1] == file_hash and existing[2] == file_stat.st_mtime:
                    skipped_count += 1
                    processed_count += 1
                    if progress_callback:
                        progress_callback(processed_count / total_files, f"Skipped: {file_path.name}")
                    continue
                
                # Process new or changed file
                content = self.read_file_content(file_path)
                
                if existing:
                    # Update existing file
                    cursor.execute('''
                        UPDATE files SET content = ?, size = ?, hash = ?, modified_time = ?
                        WHERE id = ?
                    ''', (content, len(content.encode('utf-8')), file_hash, file_stat.st_mtime, existing[0]))
                    file_id = existing[0]
                    
                    # Delete old chunks
                    cursor.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
                else:
                    # Insert new file
                    cursor.execute('''
                        INSERT INTO files (path, content, size, extension, hash, modified_time)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (str(file_path), content, len(content.encode('utf-8')), 
                          file_path.suffix.lower(), file_hash, file_stat.st_mtime))
                    file_id = cursor.lastrowid
                
                # Generate and store chunks with embeddings
                self.process_file_chunks(cursor, file_id, content, file_path.suffix.lower())
                
                processed_count += 1
                if progress_callback:
                    progress_callback(processed_count / total_files, f"Processed: {file_path.name}")
                
                # Commit periodically to avoid memory buildup
                if processed_count % 50 == 0:
                    conn.commit()
                    
            except Exception as e:
                logging.warning(f"Could not process {file_path}: {str(e)}")
                processed_count += 1
                if progress_callback:
                    progress_callback(processed_count / total_files, f"Error: {file_path.name}")
        
        conn.commit()
        conn.close()
        return processed_count, skipped_count
    
    def process_file_chunks(self, cursor, file_id, content, extension):
        """Process file into chunks and generate embeddings"""
        lines = content.split('\n')
        chunk_size = 50 if extension in ['.md', '.txt', '.pdf', '.docx'] else 25
        
        chunks_processed = 0
        for i in range(0, len(lines), chunk_size):
            chunk = '\n'.join(lines[i:i+chunk_size])
            if not chunk.strip():
                continue
            
            # Generate embedding (only if embedding model is available)
            embedding_blob = None
            if embedding_model:
                try:
                    embedding = embedding_model.encode([chunk])[0]
                    embedding_blob = pickle.dumps(embedding)
                except Exception as e:
                    logging.warning(f"Could not generate embedding: {str(e)}")
            
            cursor.execute('''
                INSERT INTO chunks (file_id, chunk_idx, content, embedding)
                VALUES (?, ?, ?, ?)
            ''', (file_id, chunks_processed, chunk, embedding_blob))
            
            chunks_processed += 1
    
    def search_similar_chunks(self, query, top_k=5, similarity_threshold=0.3, file_type_filter=None):
        """Search for similar chunks using embeddings with optimized database queries"""
        if not embedding_model:
            return []
        
        try:
            query_embedding = embedding_model.encode([query])[0]
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build query with optional file type filter
            base_query = '''
                SELECT c.content, c.file_id, c.chunk_idx, f.path, c.embedding
                FROM chunks c
                JOIN files f ON c.file_id = f.id
                WHERE c.embedding IS NOT NULL
            '''
            
            params = []
            if file_type_filter:
                placeholders = ','.join(['?' for _ in file_type_filter])
                base_query += f' AND f.extension IN ({placeholders})'
                params.extend(file_type_filter)
            
            # Limit initial results to avoid loading too much data
            base_query += ' LIMIT 2000'
            
            cursor.execute(base_query, params)
            
            results = []
            for row in cursor.fetchall():
                content, file_id, chunk_idx, path, embedding_blob = row
                
                if embedding_blob:
                    chunk_embedding = pickle.loads(embedding_blob)
                    similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
                    
                    if similarity >= similarity_threshold:
                        results.append({
                            'content': content[:500],  # Truncate content for memory efficiency
                            'full_content': content,
                            'path': path,
                            'chunk_idx': chunk_idx,
                            'similarity': similarity
                        })
            
            conn.close()
            
            # Sort by similarity and return top results
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logging.error(f"Error in similarity search: {str(e)}")
            return []
    
    def get_file_stats(self):
        """Get codebase statistics from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*), SUM(size), AVG(size) FROM files")
        result = cursor.fetchone()
        file_count, total_size, avg_size = result if result[0] else (0, 0, 0)
        
        cursor.execute("SELECT extension, COUNT(*) FROM files GROUP BY extension ORDER BY COUNT(*) DESC")
        file_types = dict(cursor.fetchall())
        
        # Get code vs doc files count
        cursor.execute("SELECT COUNT(*) FROM files WHERE extension IN ({})".format(
            ','.join(['?' for _ in SUPPORTED_CODE_EXTENSIONS])
        ), SUPPORTED_CODE_EXTENSIONS)
        code_files = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM files WHERE extension IN ({})".format(
            ','.join(['?' for _ in SUPPORTED_DOC_EXTENSIONS])
        ), SUPPORTED_DOC_EXTENSIONS)
        doc_files = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'file_count': file_count or 0,
            'total_size': total_size or 0,
            'file_types': file_types,
            'avg_file_size': avg_size or 0,
            'code_files': code_files,
            'doc_files': doc_files
        }
    
    def get_files_paginated(self, page=0, page_size=50, extension_filter=None, sort_by='size'):
        """Get files with pagination for UI display"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT path, size, extension FROM files"
        params = []
        
        if extension_filter:
            query += " WHERE extension IN ({})".format(','.join(['?' for _ in extension_filter]))
            params.extend(extension_filter)
        
        # Add sorting
        if sort_by == 'size':
            query += " ORDER BY size DESC"
        elif sort_by == 'name':
            query += " ORDER BY path"
        elif sort_by == 'extension':
            query += " ORDER BY extension, path"
        
        query += " LIMIT ? OFFSET ?"
        params.extend([page_size, page * page_size])
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        # Get total count for pagination
        count_query = "SELECT COUNT(*) FROM files"
        if extension_filter:
            count_query += " WHERE extension IN ({})".format(','.join(['?' for _ in extension_filter]))
            cursor.execute(count_query, extension_filter)
        else:
            cursor.execute(count_query)
        
        total_count = cursor.fetchone()[0]
        
        conn.close()
        
        return results, total_count
    
    def get_file_hash(self, file_path):
        """Generate hash for file to detect changes"""
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except:
            return ""
    
    def should_ignore(self, path):
        """Check if path should be ignored"""
        ignore_patterns = [
            '.git', '__pycache__', 'node_modules', 'venv', 'env', 
            '.idea', '.vscode', 'dist', 'build', '.next', 'target',
            'bin', 'obj', '.DS_Store', 'Thumbs.db', '.pytest_cache',
            'coverage', '.coverage', '.nyc_output', 'logs', '*.log'
        ]
        path_str = str(path).lower()
        return any(pattern in path_str for pattern in ignore_patterns)
    
    def cleanup_database(self):
        """Remove entries for deleted files"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, path FROM files")
        deleted_count = 0
        for file_id, path in cursor.fetchall():
            if not Path(path).exists():
                cursor.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
                cursor.execute("DELETE FROM files WHERE id = ?", (file_id,))
                deleted_count += 1
        
        conn.commit()
        conn.close()
        return deleted_count
    
    def get_database_size(self):
        """Get database file size"""
        try:
            return os.path.getsize(self.db_path)
        except:
            return 0
    
    def read_file_content(self, file_path):
        """Read file content with appropriate method based on extension"""
        ext = file_path.suffix.lower()
        
        # Handle document files
        if ext == '.pdf':
            return extract_text_from_pdf(file_path)
        elif ext in ['.docx', '.doc']:
            return extract_text_from_docx(file_path)
        elif ext in ['.xlsx', '.xls']:
            return extract_text_from_excel(file_path)
        else:
            # Handle text/code files
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            return f"Error: Could not decode file {file_path}"

# --------------------------
# Document Processing Functions (unchanged)
# --------------------------
def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    if not PDF_AVAILABLE:
        return "PDF processing not available. Install PyPDF2."
    
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    if not DOCX_AVAILABLE:
        return "DOCX processing not available. Install python-docx."
    
    try:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        return f"Error reading DOCX: {str(e)}"

def extract_text_from_excel(file_path):
    """Extract text from Excel file"""
    if not EXCEL_AVAILABLE:
        return "Excel processing not available. Install openpyxl."
    
    try:
        import pandas as pd
        df = pd.read_excel(file_path)
        return df.to_string()
    except Exception as e:
        return f"Error reading Excel: {str(e)}"

# --------------------------
# Cached Codebase Manager
# --------------------------
@st.cache_resource
def get_codebase_manager():
    """Get cached codebase manager instance"""
    return LargeCodebaseManager(st.session_state.db_path)

# --------------------------
# Enhanced UI Components
# --------------------------
def show_enhanced_codebase_panel():
    """Enhanced Codebase Management Panel for Large Codebases"""
    st.sidebar.title("Code Management:")
    
    manager = get_codebase_manager()
    
    # File upload (kept for small files/testing)
    st.sidebar.subheader("📁 Upload Files:")
    uploaded_files = st.sidebar.file_uploader(
        "Choose files or ZIP archive", 
        type=[ext[1:] for ext in SUPPORTED_EXTENSIONS] + ['zip'],
        accept_multiple_files=True,
        help="For large codebases, use directory scanning instead"
    )
    
    # Directory input with incremental scanning
    st.sidebar.subheader("📂 Directory Scanning")
    dir_path = st.sidebar.text_input(
        "Directory path:", 
        placeholder="/path/to/your/large/project",
        help="Supports incremental scanning - only processes changed files"
    )
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔍 Scan", type="primary"):
            if dir_path and os.path.isdir(dir_path):
                progress_bar = st.sidebar.progress(0)
                status_text = st.sidebar.empty()
                
                def progress_callback(progress, message):
                    progress_bar.progress(progress)
                    status_text.text(message)
                
                with st.spinner("Scanning directory incrementally..."):
                    start_time = time.time()
                    processed, skipped = manager.scan_directory_incremental(dir_path, progress_callback)
                    end_time = time.time()
                    
                    st.sidebar.success(
                        f"✅ Processed: {processed} files\n"
                        f"⏭️ Skipped: {skipped} files\n"
                        f"⏱️ Time: {end_time - start_time:.1f}s"
                    )
                
                progress_bar.empty()
                status_text.empty()
    
    with col2:
        if st.button("🧹 Clean"):
            deleted = manager.cleanup_database()
            st.sidebar.success(f"🗑️ Deleted {deleted} missing files")
    
    # Git repository cloning (enhanced for large repos)
    st.sidebar.subheader("🌐 Git Repository")
    repo_url = st.sidebar.text_input(
        "Repository URL:",
        placeholder="https://github.com/user/repo.git",
        help="Clone and automatically scan large repositories"
    )
    
    if st.sidebar.button("🔄 Clone & Scan"):
        if repo_url:
            with st.spinner("Cloning repository..."):
                temp_dir = tempfile.mkdtemp()
                repo_path = clone_repository(repo_url, temp_dir)
                if repo_path:
                    progress_bar = st.sidebar.progress(0)
                    status_text = st.sidebar.empty()
                    
                    def progress_callback(progress, message):
                        progress_bar.progress(progress)
                        status_text.text(message)
                    
                    with st.spinner("Processing repository files..."):
                        processed, skipped = manager.scan_directory_incremental(repo_path, progress_callback)
                        st.sidebar.success(f"✅ Repository loaded: {processed} files processed")
                    
                    progress_bar.empty()
                    status_text.empty()
    
    # Enhanced codebase status with database info
    st.sidebar.subheader("📊 Codebase Status")
    stats = manager.get_file_stats()
    
    if stats['file_count'] > 0:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Files", f"{stats['file_count']:,}")
            st.metric("Code Files", f"{stats['code_files']:,}")
        with col2:
            st.metric("Size (MB)", f"{stats['total_size'] / (1024*1024):.1f}")
            st.metric("Doc Files", f"{stats['doc_files']:,}")
        
        # Database size
        db_size = manager.get_database_size()
        st.sidebar.metric("DB Size (MB)", f"{db_size / (1024*1024):.1f}")
        
        # File types breakdown (top 10)
        with st.sidebar.expander("📈 Top File Types"):
            top_types = sorted(stats['file_types'].items(), key=lambda x: x[1], reverse=True)[:10]
            for ext, count in top_types:
                st.write(f"{ext}: {count:,}")
        
        # Quick stats
        avg_size_kb = stats['avg_file_size'] / 1024
        st.sidebar.caption(f"Avg file size: {avg_size_kb:.1f} KB")
        
    else:
        st.sidebar.info("No codebase loaded")
    
    # Process uploaded files (for backward compatibility)
    if uploaded_files:
        with st.spinner("Processing uploaded files..."):
            temp_dir = tempfile.mkdtemp()
            process_uploaded_files(uploaded_files, temp_dir)
            processed, skipped = manager.scan_directory_incremental(temp_dir)
            st.sidebar.success(f"✅ Uploaded files processed: {processed}")

def process_uploaded_files(uploaded_files, temp_dir):
    """Process uploaded files and extract content"""
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process ZIP files
        if uploaded_file.name.endswith('.zip'):
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                os.remove(file_path)
            except zipfile.BadZipFile:
                st.error(f"Invalid ZIP file: {uploaded_file.name}")

def clone_repository(repo_url, temp_dir):
    """Clone git repository to temporary directory"""
    try:
        repo_name = repo_url.split('/')[-1]
        if repo_name.endswith('.git'):
            repo_name = repo_name[:-4]
        repo_path = os.path.join(temp_dir, repo_name)
        
        # Clone repository
        git.Repo.clone_from(repo_url, repo_path)
        st.success(f"Repository cloned successfully to {repo_path}")
        return repo_path
    except git.exc.GitCommandError as e:
        st.error(f"Error cloning repository: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None

def show_analytics_dashboard():
    """Enhanced analytics dashboard with database queries"""
    st.header("📊 Large Codebase Analytics Dashboard")
    
    manager = get_codebase_manager()
    stats = manager.get_file_stats()
    
    if stats['file_count'] == 0:
        st.info("👆 Load a codebase using the sidebar to see analytics")
        return
    
    # Key metrics with better formatting
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Files", f"{stats['file_count']:,}")
    with col2:
        st.metric("Total Size", f"{stats['total_size'] / (1024*1024):.1f} MB")
    with col3:
        st.metric("File Types", len(stats['file_types']))
    with col4:
        st.metric("Avg File Size", f"{stats['avg_file_size'] / 1024:.1f} KB")
    
    # File type distribution with better visualization
    st.subheader("📈 File Type Distribution")
    if stats['file_types']:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Bar chart with top file types
            top_types = dict(sorted(stats['file_types'].items(), key=lambda x: x[1], reverse=True)[:15])
            fig, ax = plt.subplots(figsize=(12, 6))
            types = list(top_types.keys())
            counts = list(top_types.values())
            
            bars = ax.bar(types, counts, color='skyblue', edgecolor='navy')
            ax.set_xlabel('File Extensions')
            ax.set_ylabel('Number of Files')
            ax.set_title('Top 15 File Types by Count')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{int(height):,}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Pie chart for top types
            top_5_types = dict(list(top_types.items())[:5])
            other_count = sum(stats['file_types'].values()) - sum(top_5_types.values())
            if other_count > 0:
                top_5_types['Others'] = other_count
            
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            wedges, texts, autotexts = ax2.pie(
                top_5_types.values(), 
                labels=top_5_types.keys(), 
                autopct='%1.1f%%', 
                startangle=90
            )
            ax2.set_title('File Type Proportions (Top 5 + Others)')
            
            # Make percentage text more readable
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            st.pyplot(fig2)
    
    # File browser with advanced pagination and filtering
    st.subheader("📁 File Browser")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        page_size = st.selectbox("Files per page", [25, 50, 100, 200], index=1)
    with col2:
        sort_by = st.selectbox("Sort by", ["size", "name", "extension"])
    with col3:
        ext_options = list(stats['file_types'].keys())
        selected_exts = st.multiselect("Filter by type", ext_options)
    with col4:
        page = st.number_input("Page", min_value=0, value=0)
    
    # Get paginated files
    files, total_count = manager.get_files_paginated(
        page=page, 
        page_size=page_size, 
        extension_filter=selected_exts if selected_exts else None,
        sort_by=sort_by
    )
    
    total_pages = (total_count + page_size - 1) // page_size
    
    if files:
        st.write(f"Showing {len(files)} files (Page {page + 1} of {total_pages}, Total: {total_count:,})")
        
        # Create DataFrame for better display
        df_data = []
        for path, size, ext in files:
            df_data.append({
                'Filename': Path(path).name,
                'Extension': ext,
                'Size (KB)': f"{size / 1024:.1f}",
                'Size (MB)': f"{size / (1024*1024):.2f}",
                'Full Path': path
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, height=400)
        
        # Pagination controls
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("⬅️ Previous") and page > 0:
                st.rerun()
        with col2:
            st.write(f"Page {page + 1} of {total_pages}")
        with col3:
            if st.button("➡️ Next") and page < total_pages - 1:
                st.rerun()

def show_advanced_search():
    """Enhanced code search interface with database queries"""
    st.header("🔍 Advanced Code Search")
    
    manager = get_codebase_manager()
    stats = manager.get_file_stats()
    
    if stats['file_count'] == 0:
        st.info("👆 Load a codebase using the sidebar to enable search")
        return
    
    # Search interface with better layout
    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        query = st.text_input(
            "Search your codebase:", 
            placeholder="e.g., 'authentication function', 'API endpoint', 'error handling'",
            help="Describe what you're looking for in natural language"
        )
    with col2:
        num_results = st.selectbox("Results", [3, 5, 10, 20], index=1)
    with col3:
        search_mode = st.selectbox("Mode", ["Smart", "Fast"])
    
    # Search filters with better organization
    with st.expander("🎛️ Advanced Search Filters"):
        col1, col2, col3 = st.columns(3)
        with col1:
            file_types = st.multiselect(
                "Filter by file type:",
                options=list(stats['file_types'].keys()),
                help="Leave empty to search all file types"
            )
        with col2:
            min_similarity = st.slider(
                "Minimum similarity:",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="Higher values return more relevant results"
            )
        with col3:
            max_file_size = st.slider(
                "Max file size (KB):",
                min_value=0,
                max_value=int(stats['avg_file_size'] * 10 / 1024),
                value=int(stats['avg_file_size'] * 5 / 1024),
                help="Filter out very large files"
            )
    
    # Search execution
    if st.button("🔍 Search", type="primary") and query:
        with st.spinner("Searching large codebase..."):
            start_time = time.time()
            
            # Use database-optimized search
            results = manager.search_similar_chunks(
                query, 
                top_k=num_results * 2,  # Get more results for filtering
                similarity_threshold=min_similarity,
                file_type_filter=file_types if file_types else None
            )
            
            # Additional filtering by file size
            if max_file_size > 0:
                results = [r for r in results if len(r['content'].encode('utf-8')) <= max_file_size * 1024]
            
            # Limit to requested number of results
            results = results[:num_results]
            
            search_time = time.time() - start_time
            
            if not results:
                st.info("🚫 No matching code found. Try adjusting your search terms or filters.")
                return
            
            # Display search results with enhanced UI
            st.subheader(f"🎯 Found {len(results)} matches in {search_time:.2f}s:")
            
            for i, result in enumerate(results):
                with st.expander(
                    f"#{i+1} • {Path(result['path']).name} • Similarity: {result['similarity']:.3f}",
                    expanded=(i == 0)  # Expand first result by default
                ):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        # Show code with syntax highlighting if possible
                        file_ext = Path(result['path']).suffix[1:]  # Remove the dot
                        if file_ext in ['py', 'js', 'java', 'cpp', 'go', 'rs', 'php']:
                            st.code(result['full_content'], language=file_ext)
                        else:
                            st.code(result['full_content'], language='text')
                    with col2:
                        st.write(f"**File:** `{Path(result['path']).name}`")
                        st.write(f"**Path:** `{result['path']}`")
                        st.write(f"**Chunk:** {result['chunk_idx'] + 1}")
                        st.write(f"**Similarity:** `{result['similarity']:.3f}`")
                        st.write(f"**Size:** {len(result['full_content'])} chars")
                        
                        # Quick action buttons
                        if st.button(f"📋 Copy", key=f"copy_{i}"):
                            st.code(result['full_content'])
                            st.success("Code ready to copy!")

def find_relevant_code(query, top_k=5):
    """Find relevant code chunks using the database manager"""
    manager = get_codebase_manager()
    return manager.search_similar_chunks(query, top_k=top_k, similarity_threshold=0.2)

# --------------------------
# Main App Structure
# --------------------------
def main():
    # Check if required services are available
    if not client:
        st.error("❌ Groq client not initialized. Please check your API key.")
        return
    
    if not embedding_model:
        st.error("❌ Embedding model not loaded. Some features may not work.")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["💬 Chat Assistant", "📊 Analytics", "🔍 Code Search"])
    
    # Enhanced Codebase Management Panel
    show_enhanced_codebase_panel()
    
    # Chat Tab with optimized context retrieval
    with tab1:
        st.title("🤖 LLM Chat Assistant for Large Codebases")
        
        # Show loaded codebase context with database stats
        manager = get_codebase_manager()
        stats = manager.get_file_stats()
        
        if stats['file_count'] > 0:
            with st.expander("📁 Loaded Codebase Context", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Files Loaded", f"{stats['file_count']:,}")
                with col2:
                    st.metric("Total Size", f"{stats['total_size'] / (1024*1024):.1f} MB")
                with col3:
                    st.metric("File Types", len(stats['file_types']))
                with col4:
                    st.metric("DB Size", f"{manager.get_database_size() / (1024*1024):.1f} MB")
                
                st.info("🧠 AI has indexed your large codebase for contextual understanding!")
                
                # Quick stats about most common file types
                top_types = sorted(stats['file_types'].items(), key=lambda x: x[1], reverse=True)[:5]
                st.caption(f"Top file types: {', '.join([f'{ext}({count})' for ext, count in top_types])}")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # User input with enhanced context retrieval
        if prompt := st.chat_input("Ask coding questions, request code reviews, or get help with your large project..."):
            # Find relevant code from large codebase using database
            context_prompt = ""
            if stats['file_count'] > 0 and embedding_model:
                with st.spinner("🔍 Searching your codebase..."):
                    relevant_code = find_relevant_code(prompt, top_k=3)
                    
                if relevant_code:
                    context_prompt = "\n\n**🔍 Relevant Code from Your Project:**\n"
                    for i, code in enumerate(relevant_code):
                        # Truncate very long code chunks for context
                        content = code['full_content']
                        if len(content) > 800:
                            content = content[:400] + "\n... [truncated] ...\n" + content[-400:]
                        
                        context_prompt += f"**File: {Path(code['path']).name}** (similarity: {code['similarity']:.2f})\n"
                        context_prompt += f"```\n{content}\n```\n\n"
                    
                    context_prompt += f"*Found {len(relevant_code)} relevant code chunks from your {stats['file_count']:,} files.*\n"
            
            # Build full prompt with context
            full_prompt = f"{context_prompt}\n\n**❓ User Question:**\n{prompt}"
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                try:
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    # Create messages for the API call
                    messages = [{"role": "system", "content": "You are an expert coding assistant working with large codebases. Use the provided code context when relevant to give detailed, helpful responses. Reference specific files and code patterns when answering."}]
                    
                    # Add conversation history (keep last 10 messages to avoid token limits)
                    recent_messages = st.session_state.messages[-10:]
                    for msg in recent_messages:
                        if msg["role"] == "user":
                            # For the current message, use the full prompt with context
                            if msg["content"] == prompt:
                                messages.append({"role": "user", "content": full_prompt})
                            else:
                                messages.append(msg)
                        else:
                            messages.append(msg)
                    
                    # Stream the response
                    for response in client.chat.completions.create(
                        model=st.session_state.model,
                        messages=messages,
                        max_tokens=st.session_state.max_tokens,
                        temperature=temperature,
                        stream=True
                    ):
                        if response.choices[0].delta.content:
                            full_response += response.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")
                    st.info("Please check your Groq API key and connection.")
    
    # Analytics Tab
    with tab2:
        show_analytics_dashboard()
    
    # Search Tab
    with tab3:
        show_advanced_search()

# --------------------------
# Enhanced Sidebar Configuration
# --------------------------
with st.sidebar:
    st.title("🤖 LLM Configuration")
    
    # Model selection with descriptions
    model_descriptions = {
        "Llama3-8b": "Fast, efficient for quick tasks",
        "Llama3-70b": "Most capable, best for complex coding",
        "Mixtral-8x7b": "Good balance of speed and capability",
        "Gemma-7b": "Google's efficient model",
        "Llama2-70b": "Reliable, proven performance"
    }
    
    selected_model = st.selectbox(
        "Choose AI Model:",
        list(MODELS.keys()),
        index=1,
        help="Llama3-70b recommended for large codebase analysis"
    )
    st.session_state.model = MODELS[selected_model]
    st.caption(model_descriptions[selected_model])
    
    # Advanced parameters
    with st.expander("⚙️ Advanced Parameters"):
        st.session_state.max_tokens = st.slider(
            "Max Response Length:",
            min_value=200,
            max_value=200000,
            value=32768,
            step=256,
            help="Longer responses for detailed explanations"
        )
        
        temperature = st.slider(
            "Creativity Level:",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Higher = more creative, Lower = more focused"
        )
    
    st.divider()
    
    # Context preferences
    st.subheader("🎯 Development Focus")
    
    selected_language = st.multiselect(
        "Programming Languages:",
        options=LANGUAGES,
        default=["Python", "JavaScript"],
        help="Your preferred programming languages"
    )
    
    selected_domain = st.multiselect(
        "Development Domains:",
        options=DOMAINS,
        default=["Web Development", "Backend"],
        help="Your areas of focus"
    )
    
    st.divider()
    
    # Enhanced capabilities for large codebases
    st.subheader("🚀 Large Codebase Features")
    capabilities = [
        "🗄️ Database-backed file storage",
        "⚡ Incremental scanning",
        "🔍 Semantic code search", 
        "📊 Advanced analytics",
        "🔄 Git repository support",
        "📈 Performance optimization"
    ]
    
    for cap in capabilities:
        st.write(cap)
    
    st.divider()
    
    # Current session info with enhanced metrics
    st.subheader("📊 Session Info")
    st.write(f"**Model:** `{selected_model}`")
    
    manager = get_codebase_manager()
    stats = manager.get_file_stats()
    st.write(f"**Files Loaded:** {stats['file_count']:,}")
    st.write(f"**Total Size:** {stats['total_size'] / (1024*1024):.1f} MB")
    st.write(f"**Chat Messages:** {len(st.session_state.messages)}")
    
    if stats['file_count'] > 0:
        db_size = manager.get_database_size()
        st.write(f"**DB Size:** {db_size / (1024*1024):.1f} MB")
        st.write(f"**Efficiency:** {((stats['total_size'] / db_size) * 100):.1f}% compression")
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset DB"):
            try:
                os.remove(st.session_state.db_path)
                st.success("Database reset!")
                st.rerun()
            except:
                st.error("Could not reset database")

# Run the enhanced app
if __name__ == "__main__":
    main()
