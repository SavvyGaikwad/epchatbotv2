# Fixed Streamlit App - streamlit_app.py (Cloud Compatible)

# SQLite fix for Streamlit Cloud (skip on Windows for local development)
import sys
import platform

if platform.system() != "Windows":
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pysqlite3-binary"])
        import pysqlite3
        sys.modules['sqlite3'] = pysqlite3
    except Exception as e:
        print(f"SQLite fix failed (non-critical): {e}")

import streamlit as st
import os
from pathlib import Path
import logging
from typing import List, Dict, Any
import time
from datetime import datetime
import requests
from PIL import Image
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PyTorch environment fixes
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_OFFLINE'] = '0'

# Force CPU usage and avoid meta tensor issues
try:
    import torch
    torch.set_default_tensor_type('torch.FloatTensor')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except ImportError:
    pass

try:
    # Use the compatible imports you specified
    from langchain_chroma import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.docstore.document import Document
    import google.generativeai as genai
    
except ImportError as e:
    st.error(f"Missing dependencies: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Document Search Interface",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        border: none;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .chat-message {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .response-container {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 20px rgba(0,0,0,0.08);
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .image-container {
        border: 1px solid #e1e5e9;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #fafbfc;
    }
    
    .status-indicator {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .status-success {
        background: #d4edda;
        color: #155724;
    }
    
    .status-info {
        background: #d1ecf1;
        color: #0c5460;
    }
    
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #ddd, transparent);
        margin: 2rem 0;
    }
    
    .language-selector {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class ProfessionalChatbot:
    def __init__(self):
        self.setup_genai()
        self.setup_vector_db()
        self.initialize_session_state()
        self.setup_language_prompts()
    
    def setup_genai(self):
        """Initialize Gemini AI model"""
        try:
            # Get API key from environment variable or use the hardcoded one
            api_key = os.getenv("GEMINI_API_KEY", "AIzaSyAvzloY_NyX-yjtZb8EE_RdXPs3rPmMEso")
            
            # Debug information
            logger.info(f"Using API key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else 'SHORT_KEY'}")
            
            genai.configure(api_key=api_key)
            
            # Test the API key by making a simple request
            self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")
            
            # Test with a simple prompt to verify the API key works
            test_response = self.model.generate_content("Hello, are you working?")
            logger.info("Gemini AI model initialized and tested successfully")
            
        except Exception as e:
            st.error(f"Failed to initialize AI model. Error: {str(e)}")
            logger.error(f"Gemini AI initialization error: {e}")
            st.error("Please check your API key and ensure it's valid. You can get a new API key from: https://makersuite.google.com/app/apikey")
            st.stop()
    
    def setup_vector_db(self):
        """Initialize vector database with error handling for compatibility issues"""
        try:
            # Create embeddings with explicit device mapping and error handling
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={
                    'device': 'cpu',
                    'trust_remote_code': False
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 1  # Reduce batch size to avoid memory issues
                }
            )
            
            self.vector_db_path = "comprehensive_vector_db"
            
            if os.path.exists(self.vector_db_path):
                # Initialize Chroma with error handling
                try:
                    self.vector_db = Chroma(
                        persist_directory=self.vector_db_path, 
                        embedding_function=self.embedding_model,
                        collection_metadata={"hnsw:space": "cosine"}  # Explicit similarity metric
                    )
                    
                    # Test the database connection
                    test_results = self.vector_db.similarity_search("test", k=1)
                    
                    status_text = "✅ Knowledge base connected" if st.session_state.get('language', 'English') == 'English' else "✅ ナレッジベースに接続しました"
                    st.markdown(
                        f'<div class="status-indicator status-success">{status_text}</div>', 
                        unsafe_allow_html=True
                    )
                    logger.info(f"Vector database loaded from {self.vector_db_path}")
                    
                except Exception as db_error:
                    logger.error(f"Database connection error: {db_error}")
                    st.error("Vector database connection failed. Attempting to reinitialize...")
                    
                    # Try to reinitialize with basic settings
                    try:
                        self.vector_db = Chroma(
                            persist_directory=self.vector_db_path, 
                            embedding_function=self.embedding_model
                        )
                        st.success("Database reconnected successfully!")
                        logger.info("Vector database reconnected after error")
                    except Exception as reinit_error:
                        logger.error(f"Database reinitialization failed: {reinit_error}")
                        st.error("Failed to connect to knowledge base. Please restart the application.")
                        st.stop()
            else:
                error_text = "Knowledge base not found. Please contact system administrator." if st.session_state.get('language', 'English') == 'English' else "ナレッジベースが見つかりません。システム管理者にお問い合わせください。"
                st.error(error_text)
                logger.error(f"Vector database not found at {self.vector_db_path}")
                st.stop()
                
        except Exception as e:
            logger.error(f"Vector database initialization error: {e}")
            
            # Provide more specific error handling
            if "meta tensor" in str(e).lower():
                st.error("PyTorch compatibility issue detected. Please try restarting the application.")
                st.info("If the issue persists, this may be due to a PyTorch version compatibility problem.")
            elif "sqlite" in str(e).lower():
                st.error("Database compatibility issue. Please ensure all dependencies are properly installed.")
            else:
                error_text = "Failed to connect to knowledge base." if st.session_state.get('language', 'English') == 'English' else "ナレッジベースへの接続に失敗しました。"
                st.error(error_text)
            
            st.stop()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'processing' not in st.session_state:
            st.session_state.processing = False
        if 'language' not in st.session_state:
            st.session_state.language = 'English'
        if 'sidebar_auto_closed' not in st.session_state:
            st.session_state.sidebar_auto_closed = False
        if 'show_welcome_header' not in st.session_state:
            st.session_state.show_welcome_header = True

    def setup_language_prompts(self):
        """Setup language-specific prompts and text"""
        self.language_config = {
            'English': {
                'title': 'Eptura Asset AI',
                'subtitle': 'Smartest AI answering all your questions for managing your companys Assets',
                'input_placeholder': 'Ask me anything about your documents:',
                'ask_button': 'Ask Question',
                'new_chat': 'New Chat',
                'clear_history': '🗑 Clear Chat History',
                'recent_queries': 'Recent Queries',
                'response_header': '### Response',
                'visual_resources': '### Related Visual Resources',
                'quick_examples': '### Quick Start Examples',
                'documents_analyzed': 'Documents Analyzed',
                'content_types': 'Content Types',
                'visual_resources_metric': 'Visual Resources',
                'source_documents': 'Source Documents',
                'searching': 'Searching knowledge base...',
                'analyzing': 'Analyzing documents and generating response...',
                'no_docs_found': 'No relevant documents found. Please try rephrasing your question.',
                'enter_question': 'Please enter a question to get started.',
                'processing_error': 'An error occurred while processing your request. Please try again.',
                'examples': [
                    ("Data Management", "How to create and manage saved views?"),
                    ("Process Workflows", "What are the cycle count procedures?"),
                    ("Search & Filter", "How to use advanced filtering options?"),
                    ("User Management", "How to manage user roles and permissions?"),
                    ("Reporting", "How to generate comprehensive reports?"),
                    ("Configuration", "How to configure system settings?")
                ]
            },
            'Japanese': {
                'title': 'エプチュラ アセット AI',  # Fixed: Removed special characters that might cause encoding issues
                'subtitle': '企業アセット管理のための最もスマートなAIアシスタント',  # Improved Japanese translation
                'input_placeholder': 'ドキュメントについて何でもお聞きください：',
                'ask_button': '質問する',
                'new_chat': '新しいチャット',
                'clear_history': '🗑 履歴をクリア',
                'recent_queries': '最近の質問',
                'response_header': '### 回答',
                'visual_resources': '### 関連するビジュアルリソース',
                'quick_examples': '### クイックスタートの例',
                'documents_analyzed': '分析されたドキュメント',
                'content_types': 'コンテンツタイプ',
                'visual_resources_metric': 'ビジュアルリソース',
                'source_documents': 'ソースドキュメント',
                'searching': 'ナレッジベースを検索中...',
                'analyzing': 'ドキュメントを分析して回答を生成中...',
                'no_docs_found': '関連するドキュメントが見つかりませんでした。質問を言い換えてみてください。',
                'enter_question': '開始するには質問を入力してください。',
                'processing_error': 'リクエストの処理中にエラーが発生しました。再試行してください。',
                'examples': [
                    ("データ管理", "保存ビューの作成と管理方法は？"),
                    ("プロセスワークフロー", "サイクルカウントの手順は何ですか？"),
                    ("検索とフィルター", "高度なフィルタリングオプションの使用方法は？"),
                    ("ユーザー管理", "ユーザーの役割と権限の管理方法は？"),
                    ("レポート", "包括的なレポートの生成方法は？"),
                    ("設定", "システム設定の構成方法は？")
                ]
            }
        }

   
    def get_text(self, key):
        """Get localized text based on current language"""
        return self.language_config[st.session_state.language][key]

    @staticmethod
    def convert_github_url_to_raw(github_url):
        """Convert GitHub URL to raw format"""
        if "github.com" in github_url and "/blob/" in github_url:
            return github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        return github_url

    @staticmethod
    def is_gif_file(url):
        """Check if URL points to a GIF file"""
        return url.lower().endswith('.gif')

    def display_image_inline(self, image_url, container=None, size="small"):
        """Display image using Streamlit columns with calculated ratios for centering"""
        try:
            raw_url = self.convert_github_url_to_raw(image_url)
            display_target = container if container else st
            
            if not self.is_gif_file(raw_url):
                response = requests.get(raw_url, timeout=10)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))
                
                # Calculate display width
                width_map = {"thumbnail": 120, "small": 200, "medium": 300, "large": 400}
                width = width_map.get(size, 200)
                
                # Create columns with appropriate ratios based on image width
                # Assuming container width of ~700px, calculate ratios
                container_width = 700
                image_ratio = width / container_width
                side_ratio = (1 - image_ratio) / 2
                
                # Create columns
                left_col, center_col, right_col = display_target.columns([side_ratio, image_ratio, side_ratio])
                
                with center_col:
                    # Add container styling
                    st.markdown("""
                    <div style="
                        padding: 8px;
                        background: #f8f9fa;
                        border-radius: 6px;
                        border: 1px solid #e9ecef;
                        text-align: center;
                        margin: 8px 0;
                    ">
                    """, unsafe_allow_html=True)
                    
                    st.image(img, use_column_width=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            
            return True
        except Exception as e:
            logger.error(f"Inline image display error: {e}")
            return False
    
    def extract_content_with_media_mapping(self, docs):
        """Extract content and create mapping between text sections and their associated media"""
        content_sections = []
        
        for doc_idx, doc in enumerate(docs):
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            
            # Extract media URLs for this document
            media_urls_str = metadata.get('media_urls', '')
            associated_images = []
            if media_urls_str and media_urls_str.strip():
                associated_images = [img.strip() for img in media_urls_str.split('|') if img.strip()]
            
            # Create content section with associated media
            section_data = {
                'text_content': doc.page_content,
                'associated_media': associated_images,
                'metadata': metadata,
                'doc_index': doc_idx,
                'source': metadata.get('source', 'Unknown'),
                'document_title': metadata.get('document_title', 'Unknown'),
                'section': metadata.get('section', ''),
                'document_type': metadata.get('document_type', 'Content')
            }
            
            content_sections.append(section_data)
        
        return content_sections

    def display_response_with_inline_media(self, response_text, content_sections, container):
        """Display response with compact media embedded inline"""
        with container:
            st.markdown('<div class="response-container">', unsafe_allow_html=True)
            st.markdown(self.get_text('response_header'))
            
            # Create a mapping of all available images
            all_images = {}
            for section in content_sections:
                for img_url in section['associated_media']:
                    all_images[img_url] = section
            
            # Split response by image markers and display content with inline images
            import re
            
            # Find all image markers in the response
            image_pattern = r'\[DISPLAY_IMAGE:\s*([^\]]+)\]'
            parts = re.split(image_pattern, response_text)
            
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    # This is text content
                    if part.strip():
                        st.markdown(part.strip())
                else:
                    # This is an image URL from the marker
                    image_url = part.strip()
                    # Try to find the exact URL or a partial match
                    matching_url = None
                    for available_url in all_images.keys():
                        if image_url in available_url or available_url in image_url:
                            matching_url = available_url
                            break
                    
                    if matching_url:
                        # Display the image inline with small size for compactness
                        success = self.display_image_inline(matching_url, st, size="small")
                        if not success:
                            st.info(f"📷 {matching_url.split('/')[-1]}")
                    else:
                        # If no exact match, try to display any image that might be related
                        if all_images and image_url:
                            # Try partial matching
                            for url in all_images.keys():
                                if any(keyword in url.lower() for keyword in image_url.lower().split()):
                                    success = self.display_image_inline(url, st, size="small")
                                    if not success:
                                        st.info(f"📷 {url.split('/')[-1]}")
                                    break
            
            # If no image markers were found, display compact media gallery
            if '[DISPLAY_IMAGE:' not in response_text and all_images:
                # Compact expandable section for related images
                with st.expander(f"📷 Related Images ({len(all_images)})", expanded=False):
                    # Display images in horizontal layout with thumbnails
                    if len(all_images) <= 4:
                        cols = st.columns(len(all_images))
                    else:
                        cols = st.columns(4)
                    
                    for idx, (img_url, section) in enumerate(all_images.items()):
                        with cols[idx % len(cols)]:
                            st.markdown(f"{section['document_title'][:20]}..." if len(section['document_title']) > 20 else f"{section['document_title']}")
                            success = self.display_image_inline(img_url, st, size="thumbnail")
                            if not success:
                                st.info(f"📷 {img_url.split('/')[-1]}")
            
            st.markdown('</div>', unsafe_allow_html=True)

    def generate_response_with_inline_media_instructions(self, query, content_sections):
        """Generate AI response with instructions for compact inline media placement"""
        
        # Build enhanced context
        context_parts = []
        media_inventory = []
        
        for i, section in enumerate(content_sections):
            metadata = section['metadata']
            source = section['source']
            doc_title = section['document_title']
            section_name = section['section']
            doc_type = section['document_type']
            
            header = f"{doc_title}"
            if section_name:
                header += f" - {section_name}"
            
            # Add media information to context
            doc_content = section['text_content']
            if section['associated_media']:
                doc_content += f"\n\nAssociated Visual Resources: {', '.join(section['associated_media'])}"
                # Add to media inventory
                for media_url in section['associated_media']:
                    media_inventory.append({
                        'url': media_url,
                        'section_index': i,
                        'context': section['text_content'][:200] + "..." if len(section['text_content']) > 200 else section['text_content']
                    })
            
            context_parts.append(f"{header}\n{doc_content}")
        
        context = "\n\n---\n\n".join(context_parts)

        # Create enhanced language-specific prompt with compact media instructions
        if st.session_state.language == 'Japanese':
            prompt = f"""
プロフェッショナルなドキュメントアシスタントとして、提供されたコンテキストに基づいて包括的で正確な回答を日本語で提供してください。

重要な指示:
- 明確でプロフェッショナルな日本語を使用する
- 簡潔で読みやすい回答を提供し、不要な冗長性を避ける
- 関連する画像がある場合のみ [DISPLAY_IMAGE: image_url] マーカーを戦略的に使用する
- 画像は説明の直後ではなく、最も関連性の高い箇所にのみ表示する
- 長いテキストの間に画像を配置して読みやすさを向上させる
- 画像は最大2-3個まで選択し、最も重要なもののみを表示する
- ステップバイステップの指示には簡潔な箇条書きを使用する

利用可能な画像: {len(media_inventory)}個
画像一覧: {[item['url'] for item in media_inventory]}

ナレッジベースからのコンテキスト:
{context}

ユーザーの質問: {query}

簡潔で読みやすく、最も重要な画像のみを含む詳細な回答を日本語で提供してください:
"""
        else:
            prompt = f"""
As a professional document assistant, provide a comprehensive yet concise answer based on the provided context, strategically integrating only the most relevant visual resources inline.

Critical Instructions:
- Use clear, professional language
- Keep responses concise and scannable - avoid excessive length
- Use [DISPLAY_IMAGE: image_url] markers sparingly - only for the most relevant images (max 2-3)
- Place images strategically to break up long text sections for better readability
- Select only the most important and directly relevant images
- Use bullet points for step-by-step instructions
- Focus on readability and user experience over showing all available media

Available Images: {len(media_inventory)}
Image List: {[item['url'] for item in media_inventory]}

Context from knowledge base:
{context}

User Question: {query}

Please provide a concise, well-structured response that includes only the most relevant visual resources using [DISPLAY_IMAGE: image_url] markers strategically:
"""

        try:
            response = self.model.generate_content(prompt)
            return response.text, content_sections
        except Exception as e:
            logger.error(f"AI response generation error: {e}")
            error_text = "I apologize, but I'm unable to generate a response at this time. Please try again or contact support." if st.session_state.language == 'English' else "申し訳ございませんが、現在回答を生成できません。再試行するかサポートにお問い合わせください。"
            return error_text, []

    def process_query(self, query, container=None):
        """Enhanced query processing with inline media display and error handling"""
        try:
            # Use provided container or create a new one
            if container is None:
                container = st.container()
            
            with container:
                # Search documents with error handling
                with st.spinner(self.get_text('searching')):
                    try:
                        docs = self.vector_db.similarity_search(query, k=20)
                    except Exception as search_error:
                        logger.error(f"Search error: {search_error}")
                        st.error("Search failed. Please try again or contact support.")
                        return
                
                if not docs:
                    st.warning(self.get_text('no_docs_found'))
                    return
                
                # Extract content with media mapping
                with st.spinner(self.get_text('analyzing')):
                    try:
                        content_sections = self.extract_content_with_media_mapping(docs)
                        response_text, content_sections = self.generate_response_with_inline_media_instructions(query, content_sections)
                    except Exception as analysis_error:
                        logger.error(f"Analysis error: {analysis_error}")
                        st.error("Analysis failed. Please try again.")
                        return
                
                # Display response with inline media
                self.display_response_with_inline_media(response_text, content_sections, container)
                
                # Update chat history
                timestamp = datetime.now().strftime("%H:%M")
                total_images = sum(len(section['associated_media']) for section in content_sections)
                st.session_state.chat_history.append({
                    'timestamp': timestamp,
                    'query': query,
                    'response': response_text,
                    'doc_count': len(docs),
                    'image_count': total_images,
                    'has_media': total_images > 0
                })

                if not st.session_state.sidebar_auto_closed:
                    st.session_state.sidebar_auto_closed = True
                    # Add a small JavaScript to collapse the sidebar
                    st.markdown("""
                    <script>
                    setTimeout(function() {
                        const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
                        if (sidebar) {
                            const collapseButton = sidebar.querySelector('[data-testid="collapsedControl"]');
                            if (!collapseButton) {
                                const button = sidebar.querySelector('button[aria-label="Close sidebar"]');
                                if (button) button.click();
                            }
                        }
                    }, 1000);
                    </script>
                    """, unsafe_allow_html=True)
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            if container:
                with container:
                    st.error(self.get_text('processing_error'))
            else:
                st.error(self.get_text('processing_error'))

    def render_sidebar(self):
        """Enhanced sidebar with comprehensive media statistics"""
        with st.sidebar:
            # Language Selection at the top
            st.markdown("""
            <div class="language-selector">
                <h3>🌐 Language / 言語</h3>
            </div>
            """, unsafe_allow_html=True)
            
            language_options = {'English': '🇺🇸 English', 'Japanese': '🇯🇵 日本語'}
            
            new_language = st.selectbox(
                "",
                options=list(language_options.keys()),
                format_func=lambda x: language_options[x],
                index=0 if st.session_state.language == 'English' else 1,
                key="language_selector"
            )
            
            if new_language != st.session_state.language:
                st.session_state.language = new_language
                st.rerun()
            
            st.markdown("---")
            
            # Enhanced knowledge base stats
            # Knowledge base connection status only
            if hasattr(self, 'vector_db'):
                try:
                    # Simple connection test
                    test_results = self.vector_db.similarity_search("test", k=1)
                    st.markdown("---")
                except Exception as e:
                    logger.warning(f"Could not connect to knowledge base: {e}")
                    
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🆕 New Chat", key="new_chat_btn"):
                    self.reset_to_welcome_screen()

            with col2:
                if st.button("🗑 Clear", key="clear_history_btn"):
                    st.session_state.chat_history = []
                    st.rerun()
            
            # Enhanced chat history
            # Simplified chat history - queries only
            # Simple chat history - just queries
            if st.session_state.chat_history:
                st.markdown(f"### {self.get_text('recent_queries')}")
                for chat in reversed(st.session_state.chat_history[-5:]):
                    query_label = "*Query:" if st.session_state.language == 'English' else "質問:*"
                    st.write(f"{query_label} {chat['query']}")

# REPLACE the title display section in render_main_interface method (around line 520):
# Find this part and replace with the code below:

    def render_main_interface(self):
        """Enhanced main interface with dynamic title that disappears after first query"""
        
        # Show welcome header only if no queries have been made
        if st.session_state.show_welcome_header:
            # Get current language title and subtitle
            current_title = self.get_text('title')
            current_subtitle = self.get_text('subtitle')
            
            st.markdown(f"""
            <div style="text-align: center; padding: 3rem 0 2rem 0; margin-bottom: 2rem;">
                <h1 style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                        font-size: 3.5rem; font-weight: 800; margin-bottom: 1rem; line-height: 1.2;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;">
                    {current_title}
                </h1>
                <p style="color: #6c757d; font-size: 1.3rem; max-width: 800px; margin: 0 auto; line-height: 1.4;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;">
                    {current_subtitle}
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Show compact header after first query
            current_title = self.get_text('title')
            
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem 0 0.5rem 0; margin-bottom: 1rem;">
                <h2 style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                        font-size: 1.8rem; font-weight: 700; margin-bottom: 0.5rem;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;">
                    {current_title}
                </h2>
            </div>
            """, unsafe_allow_html=True)
        
    # Rest of your render_main_interface method continues here...
        
        # Render sidebar
        self.render_sidebar()
        
        # Main query interface with Enter key support
        with st.container():
            # Use a form to enable Enter key submission
            with st.form(key="query_form", clear_on_submit=False):
                query = st.text_input(
                    "",
                    placeholder=self.get_text('input_placeholder'),
                    key="main_query"
                )
                
                submit_clicked = st.form_submit_button(self.get_text('ask_button'), type="primary")

        
        # Response area container
        response_container = st.container()
        
        # Process query from input box (works with both Enter key and button click)
        if submit_clicked and query.strip():
            # Hide welcome header immediately when query is submitted
            st.session_state.show_welcome_header = False
            self.process_query(query, response_container)
        elif submit_clicked:
            with response_container:
                st.warning(self.get_text('enter_question'))
        
        # Show examples only when welcome header is visible
        if st.session_state.show_welcome_header:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            
            # Expanded examples section for welcome screen
            st.markdown(f"### {self.get_text('quick_examples').replace('### ', '')}")
            examples = self.get_text('examples')
            
            # Check if any example button was clicked
            example_clicked = None
            cols = st.columns(2)
            for idx, (category, example) in enumerate(examples[:6]):  # Limit to first 6 examples
                with cols[idx % 2]:
                    if st.button(f"{category}", key=f"example_{idx}", help=example):
                        example_clicked = example
            
            # Process example query in the response container
            if example_clicked:
                # Hide welcome header immediately when example is clicked
                st.session_state.show_welcome_header = False
                self.process_query(example_clicked, response_container)
        # 5. ADD this method to handle new chat functionality (optional enhancement)
        # ADD this new method after the render_main_interface method:

    def reset_to_welcome_screen(self):
            """Reset the interface to show the welcome screen"""
            st.session_state.show_welcome_header = True
            st.session_state.chat_history = []
            st.rerun()


def main():
    """Main application entry point"""
    try:
        chatbot = ProfessionalChatbot()
        chatbot.render_main_interface()
    except Exception as e:
        error_text = "Failed to initialize the application. Please contact support." if st.session_state.get('language', 'English') == 'English' else "アプリケーションの初期化に失敗しました。サポートにお問い合わせください。"
        st.error(error_text)
        logger.error(f"Application initialization error: {e}")

if __name__ == "__main__":
    main()
