import json
import os
import shutil
import glob
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

def load_json(filepath):
    """Load JSON file with error handling"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def flatten_media_list(media_item):
    """
    Flatten media items that might be nested lists or strings
    """
    if isinstance(media_item, str):
        return [media_item]
    elif isinstance(media_item, list):
        flattened = []
        for item in media_item:
            flattened.extend(flatten_media_list(item))
        return flattened
    else:
        # Handle other types by converting to string
        return [str(media_item)]

def process_json_content(data, source_file):
    """
    Unified processing function for all JSON document types.
    Handles glossary, FAQ, and general content with media support.
    """
    docs = []
    document_title = data.get("document_title", source_file)
    
    # Determine document type based on title or filename
    doc_type = "user_guide"  # default
    if "glossary" in document_title.lower() or source_file.lower().startswith("glossary"):
        doc_type = "glossary"
    elif "faq" in document_title.lower() or source_file.lower().startswith("faq"):
        doc_type = "faq"
    
    print(f"Processing as: {doc_type}")
    
    for section in data.get("content", []):
        section_title = section.get("title", "")
        
        if doc_type == "glossary":
            # Process glossary terms
            for item in section.get("content", []):
                if item.get("type") == "term":
                    title = item.get("title", "")
                    definition = item.get("definition", "")
                    
                    # Create full content
                    content = f"Term: {title}\nDefinition: {definition}"
                    
                    # Extract term name without quotes/abbreviations for better searchability
                    clean_title = title.split('"')[0].strip() if '"' in title else title
                    
                    metadata = {
                        "source": source_file,
                        "document_type": "glossary",
                        "document_title": document_title,
                        "section": section_title,
                        "term": clean_title,
                        "full_term": title,
                        "type": "definition",
                        "has_media": False,
                        "media_count": 0,
                        "media_urls": ""
                    }
                    
                    docs.append(Document(page_content=content, metadata=metadata))
                    
                    # Also create a searchable version with just the definition
                    docs.append(Document(
                        page_content=f"{clean_title}: {definition}",
                        metadata={**metadata, "type": "term_definition"}
                    ))
        
        elif doc_type == "faq":
            # Process FAQ questions and answers
            current_question = ""
            current_answers = []
            current_links = []
            
            for item in section.get("content", []):
                if item.get("type") == "question":
                    # Process previous Q&A if exists
                    if current_question and current_answers:
                        content = f"Question: {current_question}\nAnswer: {' '.join(current_answers)}"
                        if current_links:
                            content += f"\nReferences: {', '.join(current_links)}"
                        
                        metadata = {
                            "source": source_file,
                            "document_type": "faq",
                            "document_title": document_title,
                            "section": section_title,
                            "question": current_question,
                            "type": "qa_pair",
                            "has_links": len(current_links) > 0,
                            "has_media": False,
                            "media_count": 0,
                            "media_urls": ""
                        }
                        
                        docs.append(Document(page_content=content, metadata=metadata))
                    
                    # Start new Q&A
                    current_question = item.get("text", "")
                    current_answers = []
                    current_links = []
                    
                elif item.get("type") == "answer":
                    answer_text = item.get("text", "")
                    current_answers.append(answer_text)
                    
                    # Extract links if present
                    links = item.get("links", [])
                    current_links.extend(links)
            
            # Process final Q&A in section
            if current_question and current_answers:
                content = f"Question: {current_question}\nAnswer: {' '.join(current_answers)}"
                if current_links:
                    content += f"\nReferences: {', '.join(current_links)}"
                
                metadata = {
                    "source": source_file,
                    "document_type": "faq",
                    "document_title": document_title,
                    "section": section_title,
                    "question": current_question,
                    "type": "qa_pair",
                    "has_links": len(current_links) > 0,
                    "has_media": False,
                    "media_count": 0,
                    "media_urls": ""
                }
                
                docs.append(Document(page_content=content, metadata=metadata))
        
        else:
            # Process general content (user guides) with enhanced media handling
            if section.get("type") == "section":
                section_content = []
                section_steps = []
                section_images = []
                current_step = None
                current_substeps = []
                
                # Track content blocks with their associated media
                content_blocks = []
                current_block = {"content": [], "media": []}
                
                section_items = section.get("content", [])
                
                for i, item in enumerate(section_items):
                    item_type = item.get("type", "")
                    text = item.get("text", "")
                    
                    if item_type == "info":
                        # FIXED: Check for "image_path" (singular) instead of "image_paths" (plural)
                        if "image_path" in item:
                            # Handle image_path that might be a string or list
                            image_paths = flatten_media_list(item["image_path"])
                            current_block["media"].extend(image_paths)
                        
                        if text:
                            current_block["content"].append(f"Info: {text}")
                        
                        # Check if next item is media - if so, don't close the block yet
                        next_item = section_items[i+1] if i+1 < len(section_items) else None
                        if not next_item or next_item.get("type") != "media":
                            # Finalize current block
                            if current_block["content"] or current_block["media"]:
                                content_blocks.append(current_block.copy())
                            current_block = {"content": [], "media": []}
                    
                    elif item_type == "step":
                        # FIXED: Also check for "image_path" in steps
                        if "image_path" in item:
                            image_paths = flatten_media_list(item["image_path"])
                            current_block["media"].extend(image_paths)
                            section_images.extend(image_paths)
                        
                        # Save previous step if exists
                        if current_step:
                            if current_substeps:
                                full_step = f"{current_step}\nSubsteps: {' | '.join(current_substeps)}"
                                section_steps.append(full_step)
                            else:
                                section_steps.append(current_step)
                        
                        # Start new step
                        current_step = f"Step: {text}"
                        current_substeps = []
                    
                    elif item_type == "substep":
                        # FIXED: Also check for "image_path" in substeps
                        if "image_path" in item:
                            image_paths = flatten_media_list(item["image_path"])
                            current_block["media"].extend(image_paths)
                            section_images.extend(image_paths)
                        
                        if text:
                            current_substeps.append(text)
                    
                    elif item_type == "media":
                        # Add media to current block - handle potential list/string issue
                        media_path = item.get("path", "")
                        if media_path:
                            # Ensure media_path is treated as a string
                            media_paths = flatten_media_list(media_path)
                            current_block["media"].extend(media_paths)
                            section_images.extend(media_paths)
                    
                    # Handle any other text content that might have image_path
                    elif text:
                        # FIXED: Check for "image_path" in any content type
                        if "image_path" in item:
                            image_paths = flatten_media_list(item["image_path"])
                            current_block["media"].extend(image_paths)
                            section_images.extend(image_paths)
                        
                        if item_type not in ["step", "substep", "info", "media"]:
                            current_block["content"].append(f"{item_type}: {text}")
                
                # Don't forget the last step
                if current_step:
                    if current_substeps:
                        full_step = f"{current_step}\nSubsteps: {' | '.join(current_substeps)}"
                        section_steps.append(full_step)
                    else:
                        section_steps.append(current_step)
                
                # Don't forget the last content block
                if current_block["content"] or current_block["media"]:
                    content_blocks.append(current_block)
                
                # Create documents for content blocks with associated media
                for block_idx, block in enumerate(content_blocks):
                    if block["content"]:
                        block_content = "\n".join(block["content"])
                        # Ensure all media items are strings
                        media_list = []
                        for media_item in block["media"]:
                            media_list.extend(flatten_media_list(media_item))
                        
                        # Create content with media references
                        full_content = f"Document: {document_title}\nSection: {section_title}\n\n{block_content}"
                        if media_list:
                            full_content += f"\n\nAssociated Media:\n" + "\n".join([f"- {media}" for media in media_list])
                        
                        metadata = {
                            "source": source_file,
                            "document_type": "user_guide",
                            "document_title": document_title,
                            "section": section_title,
                            "type": "content_block",
                            "block_index": block_idx,
                            "media_urls": "|".join(media_list) if media_list else "",
                            "has_media": len(media_list) > 0,
                            "media_count": len(media_list)
                        }
                        
                        docs.append(Document(page_content=full_content, metadata=metadata))
                        
                        # Create individual searchable documents for each content item in the block
                        for content_idx, content_item in enumerate(block["content"]):
                            item_metadata = {
                                "source": source_file,
                                "document_type": "user_guide",
                                "document_title": document_title,
                                "section": section_title,
                                "block_index": block_idx,
                                "content_index": content_idx,
                                "type": "content_item",
                                "media_urls": "|".join(media_list) if media_list else "",
                                "has_media": len(media_list) > 0,
                                "media_count": len(media_list)
                            }
                            
                            item_content = f"{document_title} - {section_title}: {content_item}"
                            if media_list:
                                item_content += f"\nRelated Media: {', '.join(media_list)}"
                            
                            docs.append(Document(
                                page_content=item_content,
                                metadata=item_metadata
                            ))
                
                # Create documents for steps (may also have associated media)
                if section_steps:
                    steps_content = f"Document: {document_title}\nSection: {section_title}\n\nSteps:\n" + "\n".join(section_steps)
                    
                    # Ensure all section images are strings
                    flattened_images = []
                    for img in section_images:
                        flattened_images.extend(flatten_media_list(img))
                    
                    metadata = {
                        "source": source_file,
                        "document_type": "user_guide",
                        "document_title": document_title,
                        "section": section_title,
                        "type": "steps_section",
                        "step_count": len(section_steps),
                        "media_urls": "|".join(flattened_images) if flattened_images else "",
                        "has_media": len(flattened_images) > 0,
                        "media_count": len(flattened_images)
                    }
                    
                    docs.append(Document(page_content=steps_content, metadata=metadata))
                
                # Create a comprehensive section document
                all_content = []
                for block in content_blocks:
                    all_content.extend(block["content"])
                all_content.extend(section_steps)
                
                # Ensure all section images are strings
                flattened_images = []
                for img in section_images:
                    flattened_images.extend(flatten_media_list(img))
                
                if all_content or flattened_images:
                    full_section_content = f"Document: {document_title}\nSection: {section_title}\n\n"
                    if all_content:
                        full_section_content += "\n".join(all_content)
                    
                    if flattened_images:
                        full_section_content += f"\n\nSection Media:\n" + "\n".join([f"- {img}" for img in flattened_images])
                    
                    metadata = {
                        "source": source_file,
                        "document_type": "user_guide",
                        "document_title": document_title,
                        "section": section_title,
                        "type": "full_section",
                        "media_urls": "|".join(flattened_images) if flattened_images else "",
                        "has_media": len(flattened_images) > 0,
                        "content_count": len(all_content),
                        "media_count": len(flattened_images)
                    }
                    
                    docs.append(Document(page_content=full_section_content, metadata=metadata))
    
    return docs

def process_single_file(filepath):
    """Process a single JSON file and return documents"""
    print(f"\nProcessing: {filepath}")
    
    data = load_json(filepath)
    if not data:
        return []
    
    source_file = os.path.basename(filepath)
    document_title = data.get("document_title", source_file)
    
    print(f"Document title: {document_title}")
    
    # Use unified processing function
    docs = process_json_content(data, source_file)
    print(f"Created {len(docs)} documents")
    
    return docs

def find_all_json_files(root_folder):
    """Recursively find all JSON files in the folder and subfolders"""
    json_files = []
    
    # Use pathlib for better path handling
    root_path = Path(root_folder)
    
    if not root_path.exists():
        print(f"Folder not found: {root_folder}")
        return []
    
    # Find all JSON files recursively
    for json_file in root_path.rglob("*.json"):
        json_files.append(str(json_file))
    
    return sorted(json_files)

def create_comprehensive_vector_db(data_folder, persist_dir):
    """Create vector database from all JSON files in data folder and subfolders"""
    
    # Remove existing database
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        print(f"Removed existing database at {persist_dir}")
    
    # Find all JSON files recursively
    json_files = find_all_json_files(data_folder)
    
    if not json_files:
        print(f"No JSON files found in {data_folder}")
        return
    
    print(f"Found {len(json_files)} JSON files:")
    for file in json_files:
        relative_path = os.path.relpath(file, data_folder)
        print(f"  - {relative_path}")
    
    all_docs = []
    file_stats = {}
    
    # Process each file
    for filepath in json_files:
        try:
            docs = process_single_file(filepath)
            all_docs.extend(docs)
            
            # Track stats per file
            relative_path = os.path.relpath(filepath, data_folder)
            file_stats[relative_path] = len(docs)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue
    
    print(f"\nTotal documents created: {len(all_docs)}")
    
    if not all_docs:
        print("No documents to process!")
        return
    
    # Show file processing stats
    print("\nDocuments per file:")
    for file_path, count in file_stats.items():
        print(f"  {file_path}: {count} documents")
    
    # Show document type distribution
    doc_types = {}
    doc_titles = {}
    media_stats = {"with_media": 0, "without_media": 0}
    
    for doc in all_docs:
        doc_type = doc.metadata.get("document_type", "unknown")
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        # Track document titles
        doc_title = doc.metadata.get("document_title", doc.metadata.get("source", "unknown"))
        doc_titles[doc_title] = doc_titles.get(doc_title, 0) + 1
        
        # Track media stats
        if doc.metadata.get("has_media", False):
            media_stats["with_media"] += 1
        else:
            media_stats["without_media"] += 1
    
    print("\nDocument type distribution:")
    for doc_type, count in doc_types.items():
        print(f"  {doc_type}: {count} documents")
    
    print("\nDocument title distribution:")
    for doc_title, count in sorted(doc_titles.items()):
        print(f"  {doc_title}: {count} documents")
    
    print(f"\nMedia distribution:")
    print(f"  Documents with media: {media_stats['with_media']}")
    print(f"  Documents without media: {media_stats['without_media']}")
    
    # Create vector database
    print(f"\nCreating vector database...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(all_docs, embeddings, persist_directory=persist_dir)
    db.persist()
    print(f"Saved vector DB to {persist_dir}")
    
    return db

def test_comprehensive_db(db_path):
    """Test the comprehensive database with various queries, focusing on media-related content"""
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return
    
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    # Test queries for different document types, including media-specific ones
    test_queries = [
        # Media-related queries
        ("QR code information", "user_guide"),
        ("barcode displays asset", "user_guide"),
        ("Asset ID and Description", "user_guide"),
        ("Part barcode information", "user_guide"),
        
        # General content queries
        ("cycle count", "user_guide"),
        ("saved view", "user_guide"),
        ("column chooser", "user_guide"),
        ("recount", "user_guide"),
        
        # Installation and setup
        ("download barcode client", "user_guide"),
        ("install wizard", "user_guide"),
        
        # Glossary queries (if any)
        ("API", "glossary"),
        ("Asset Management", "glossary"),
        
        # FAQ queries (if any)
        ("mobile app", "faq"),
        ("licensing", "faq"),
        
        # General searches
        ("filter column", None),
        ("save view", None),
        ("needs recount", None)
    ]
    
    print(f"\n{'='*80}")
    print("TESTING COMPREHENSIVE DATABASE WITH UNIFIED PROCESSING")
    print('='*80)
    
    for query, expected_type in test_queries:
        print(f"\nQuery: '{query}'", end="")
        if expected_type:
            print(f" (expecting {expected_type})")
        else:
            print(" (any type)")
        print("-" * 50)
        
        results = db.similarity_search(query, k=3)
        
        if not results:
            print("  No results found")
            continue
        
        for i, result in enumerate(results, 1):
            doc_type = result.metadata.get("document_type", "unknown")
            source = result.metadata.get("source", "unknown")
            doc_title = result.metadata.get("document_title", "")
            section = result.metadata.get("section", "")
            
            print(f"  Result {i}: [{doc_type}] {source}")
            if doc_title and doc_title != source:
                print(f"    Document: {doc_title}")
            if section:
                print(f"    Section: {section}")
            print(f"    Content: {result.page_content[:150]}...")
            
            # Show media information
            if result.metadata.get("has_media"):
                media_count = result.metadata.get("media_count", 0)
                media_urls = result.metadata.get("media_urls", "")
                print(f"    Media: {media_count} item(s)")
                if media_urls:
                    media_list = media_urls.split('|')
                    for media_url in media_list[:2]:  # Show first 2 media items
                        print(f"      - {media_url}")
                    if len(media_list) > 2:
                        print(f"      ... and {len(media_list) - 2} more")
            
            print()

def query_with_filters(db_path, query, document_type=None, document_title=None, k=5):
    """
    Enhanced query function with optional filtering and media support
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    # Basic similarity search
    results = db.similarity_search(query, k=k*3)  # Get more to filter
    
    # Apply filters
    filtered_results = []
    for result in results:
        include = True
        
        if document_type and result.metadata.get("document_type") != document_type:
            include = False
        
        if document_title and document_title.lower() not in result.metadata.get("document_title", "").lower():
            include = False
        
        if include:
            filtered_results.append(result)
        
        if len(filtered_results) >= k:
            break
    
    response_data = []
    
    for i, result in enumerate(filtered_results):
        # Parse media list from media_urls string
        media_urls = result.metadata.get("media_urls", "")
        media_list = media_urls.split('|') if media_urls else []
        
        item = {
            "rank": i + 1,
            "content": result.page_content,
            "document_type": result.metadata.get("document_type", "unknown"),
            "source": result.metadata.get("source", "unknown"),
            "document_title": result.metadata.get("document_title", ""),
            "section": result.metadata.get("section", ""),
            "type": result.metadata.get("type", ""),
            "media_list": media_list,
            "media_urls": result.metadata.get("media_urls", ""),
            "has_media": result.metadata.get("has_media", False),
            "media_count": result.metadata.get("media_count", 0)
        }
        response_data.append(item)
    
    return response_data

def query_with_media_context(db_path, query, k=5):
    """
    Special query function that prioritizes results with media and provides media context
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    # Get more results initially
    results = db.similarity_search(query, k=k*2)
    
    # Separate results with and without media
    results_with_media = []
    results_without_media = []
    
    for result in results:
        if result.metadata.get("has_media", False):
            results_with_media.append(result)
        else:
            results_without_media.append(result)
    
    # Combine results, prioritizing those with media
    prioritized_results = results_with_media[:k//2] + results_without_media[:k-len(results_with_media[:k//2])]
    
    response_data = []
    
    for i, result in enumerate(prioritized_results[:k]):
        # Parse media list from media_urls string
        media_urls = result.metadata.get("media_urls", "")
        media_list = media_urls.split('|') if media_urls else []
        
        item = {
            "rank": i + 1,
            "content": result.page_content,
            "document_type": result.metadata.get("document_type", "unknown"),
            "source": result.metadata.get("source", "unknown"),
            "document_title": result.metadata.get("document_title", ""),
            "section": result.metadata.get("section", ""),
            "type": result.metadata.get("type", ""),
            "media_list": media_list,
            "has_media": result.metadata.get("has_media", False),
            "media_count": result.metadata.get("media_count", 0),
            "priority": "high" if result.metadata.get("has_media", False) else "normal"
        }
        response_data.append(item)
    
    return response_data
if __name__ == "__main__":
    # Updated path to your specified directory
    data_folder = r"C:\Users\hp\OneDrive\Desktop\epchatbot-finalvr-main\data"
    db_path = "comprehensive_vector_db"
    
    # Create the comprehensive vector database
    print("Creating comprehensive vector database with unified processing...")
    create_comprehensive_vector_db(data_folder, db_path)
    
    # Test the database
    print(f"\n{'='*80}")
    print("TESTING THE DATABASE")
    print('='*80)
    test_comprehensive_db(db_path)
    
    # Example queries with media focus
    print(f"\n{'='*80}")
    print("EXAMPLE QUERIES WITH MEDIA SUPPORT")
    print('='*80)
    
    if os.path.exists(db_path):
        example_queries = [
            ("QR Code displays Asset ID", None),
            ("barcode information displays", None),
            ("How to create a saved view?", None),
            ("install barcode client wizard", "user_guide")
        ]
        
        for query, doc_type in example_queries:
            print(f"\nQuery: '{query}'")
            if doc_type:
                print(f"Filter: {doc_type} only")
            print("-" * 40)
            
            # Use media-aware query
            results = query_with_media_context(db_path, query, k=2)
            
            for item in results:
                print(f"  [{item['document_type']}] {item['document_title']} (Priority: {item.get('priority', 'normal')})")
                if item['section']:
                    print(f"  Section: {item['section']}")
                print(f"  {item['content'][:100]}...")
                
                if item['has_media']:
                    print(f"  📷 Media ({item['media_count']} items):")
                    for media_url in item['media_list'][:2]:  # Show first 2
                        print(f"    - {media_url}")
                    if len(item['media_list']) > 2:
                        print(f"    ... and {len(item['media_list']) - 2} more")
                
                print()