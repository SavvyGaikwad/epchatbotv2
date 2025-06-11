import json
import copy

def process_content_items(content_list):
    """
    Process a list of content items and merge step/substep+media patterns into info+image_path structure
    """
    processed_items = []
    i = 0
    
    while i < len(content_list):
        current_item = content_list[i]
        current_type = current_item.get("type", "")
        
        # Check if current item is a step/substep that might be followed by media item(s)
        if current_type in ["step", "substep"]:
            step_text = current_item.get("text", "")
            media_paths = []
            
            # Look ahead to collect consecutive media items
            j = i + 1
            while (j < len(content_list) and 
                   content_list[j].get("type") == "media"):
                media_path = content_list[j].get("path", "")
                if media_path:
                    media_paths.append(media_path)
                j += 1
            
            # If we found media items, create merged info item
            if media_paths:
                new_item = {
                    "type": "info",
                    "text": step_text
                }
                
                # Add image_path as single string or array based on count
                if len(media_paths) == 1:
                    new_item["image_path"] = media_paths[0]
                else:
                    new_item["image_path"] = media_paths
                
                processed_items.append(new_item)
                # Skip all the media items we've processed
                i = j
            else:
                # No media items found, keep step/substep as is but convert to info
                new_item = {
                    "type": "info",
                    "text": step_text
                }
                processed_items.append(new_item)
                i += 1
                
        # Check if current item is a standalone media item (not preceded by step/substep)
        elif current_type == "media":
            # Check if previous item was not a step/substep (to avoid double processing)
            if (i == 0 or 
                processed_items[-1].get("type") == "info" and "image_path" not in processed_items[-1]):
                # This is a standalone media item, convert to info with image_path
                media_path = current_item.get("path", "")
                if media_path:
                    # Look for consecutive media items
                    media_paths = [media_path]
                    j = i + 1
                    while (j < len(content_list) and 
                           content_list[j].get("type") == "media"):
                        next_path = content_list[j].get("path", "")
                        if next_path:
                            media_paths.append(next_path)
                        j += 1
                    
                    new_item = {
                        "type": "info",
                        "text": "View the image below:",  # Default text for standalone media
                    }
                    
                    if len(media_paths) == 1:
                        new_item["image_path"] = media_paths[0]
                    else:
                        new_item["image_path"] = media_paths
                    
                    processed_items.append(new_item)
                    i = j
                else:
                    i += 1
            else:
                # This media item was already processed with a previous step/substep
                i += 1
                
        else:
            # Not a step/substep/media pattern, keep as is but process nested content if exists
            new_item = copy.deepcopy(current_item)
            
            # Recursively process nested content
            if "content" in new_item and isinstance(new_item["content"], list):
                new_item["content"] = process_content_items(new_item["content"])
            
            processed_items.append(new_item)
            i += 1
    
    return processed_items

def convert_json_structure(input_file, output_file):
    """
    Main function to convert JSON structure from step+media to info+image_path
    """
    try:
        # Read the input JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process the content
        if "content" in data and isinstance(data["content"], list):
            data["content"] = process_content_items(data["content"])
        
        # Write the processed JSON to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        print(f"Successfully converted {input_file} to {output_file}")
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file. {str(e)}")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def analyze_patterns(input_file):
    """
    Analyze the JSON file to find all step+media and media patterns
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        patterns_found = {
            "step_media": 0,
            "substep_media": 0,
            "standalone_media": 0,
            "step_only": 0,
            "substep_only": 0
        }
        
        def analyze_content_list(content_list, path=""):
            for i, item in enumerate(content_list):
                current_path = f"{path}[{i}]" if path else f"item[{i}]"
                item_type = item.get("type", "")
                
                if item_type == "step":
                    # Check if followed by media
                    if (i + 1 < len(content_list) and 
                        content_list[i + 1].get("type") == "media"):
                        patterns_found["step_media"] += 1
                        print(f"STEP+MEDIA found at {current_path}: {item.get('text', '')[:50]}...")
                    else:
                        patterns_found["step_only"] += 1
                        
                elif item_type == "substep":
                    # Check if followed by media
                    if (i + 1 < len(content_list) and 
                        content_list[i + 1].get("type") == "media"):
                        patterns_found["substep_media"] += 1
                        print(f"SUBSTEP+MEDIA found at {current_path}: {item.get('text', '')[:50]}...")
                    else:
                        patterns_found["substep_only"] += 1
                        
                elif item_type == "media":
                    # Check if this is standalone (not preceded by step/substep)
                    if (i == 0 or 
                        content_list[i-1].get("type") not in ["step", "substep"]):
                        patterns_found["standalone_media"] += 1
                        print(f"STANDALONE MEDIA found at {current_path}: {item.get('path', '')}")
                
                # Recursively analyze nested content
                if "content" in item and isinstance(item["content"], list):
                    analyze_content_list(item["content"], f"{current_path}.content")
        
        if "content" in data and isinstance(data["content"], list):
            analyze_content_list(data["content"])
        
        print("\nPattern Analysis Summary:")
        print("=" * 40)
        for pattern, count in patterns_found.items():
            print(f"{pattern.replace('_', ' ').title()}: {count}")
        
        total_conversions = (patterns_found["step_media"] + 
                           patterns_found["substep_media"] + 
                           patterns_found["standalone_media"])
        print(f"\nTotal items to convert: {total_conversions}")
        
        return patterns_found
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return None

def preview_changes(input_file, num_examples=5):
    """
    Preview the changes that would be made without modifying the original file
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        original_data = copy.deepcopy(data)
        
        # Process the content
        if "content" in data and isinstance(data["content"], list):
            data["content"] = process_content_items(data["content"])
        
        # Find and display examples of changes
        changes_found = 0
        print("\nPreview of changes:")
        print("=" * 50)
        
        def find_changes(original_list, processed_list, path=""):
            nonlocal changes_found
            if changes_found >= num_examples:
                return
                
            # Handle different list lengths
            min_len = min(len(original_list), len(processed_list))
            
            orig_idx = 0
            proc_idx = 0
            
            while orig_idx < len(original_list) and proc_idx < len(processed_list) and changes_found < num_examples:
                orig = original_list[orig_idx]
                proc = processed_list[proc_idx]
                
                current_path = f"{path}[{orig_idx}→{proc_idx}]" if path else f"item[{orig_idx}→{proc_idx}]"
                
                # Check if this represents a conversion
                if ((orig.get("type") in ["step", "substep"] and proc.get("type") == "info") or
                    (orig.get("type") == "media" and proc.get("type") == "info" and "image_path" in proc)):
                    
                    changes_found += 1
                    print(f"\nChange {changes_found}:")
                    print(f"Location: {current_path}")
                    print("BEFORE:")
                    print(f"  Type: {orig.get('type')}")
                    if orig.get("type") == "media":
                        print(f"  Path: {orig.get('path', '')}")
                    else:
                        print(f"  Text: {orig.get('text', '')[:80]}...")
                    
                    print("AFTER:")
                    print(f"  Type: {proc.get('type')}")
                    print(f"  Text: {proc.get('text', '')[:80]}...")
                    if "image_path" in proc:
                        if isinstance(proc.get('image_path'), list):
                            print(f"  Image paths: {len(proc['image_path'])} images")
                            for idx, img_path in enumerate(proc['image_path'][:2]):
                                print(f"    {idx+1}. {img_path}")
                            if len(proc['image_path']) > 2:
                                print(f"    ... and {len(proc['image_path']) - 2} more")
                        else:
                            print(f"  Image path: {proc.get('image_path', '')}")
                
                # Handle nested content
                if ("content" in orig and isinstance(orig["content"], list) and
                    "content" in proc and isinstance(proc["content"], list)):
                    find_changes(orig["content"], proc["content"], f"{current_path}.content")
                
                orig_idx += 1
                proc_idx += 1
        
        if "content" in original_data and "content" in data:
            find_changes(original_data["content"], data["content"])
        
        if changes_found == 0:
            print("No patterns found to convert.")
        
        return True
        
    except Exception as e:
        print(f"Error during preview: {str(e)}")
        return False

# Example usage
if __name__ == "__main__":
    input_filename = "Asset Import Template Data.json"  # Your input file
    output_filename = "Asset Import Template Data converted.json"  # Output file
    
    print("Enhanced JSON Structure Converter")
    print("Converts step/substep+media patterns to info+image_path structure")
    print("-" * 70)
    
    # First, analyze the patterns
    print("Analyzing patterns in the JSON file...")
    patterns = analyze_patterns(input_filename)
    
    if patterns and sum(patterns.values()) > 0:
        print("\n" + "=" * 70)
        
        # Preview changes
        print("Previewing changes...")
        preview_changes(input_filename, num_examples=10)
        
        print("\n" + "=" * 70)
        
        # Ask for confirmation
        response = input("Proceed with conversion? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            # Perform the conversion
            success = convert_json_structure(input_filename, output_filename)
            
            if success:
                print(f"\nConversion completed successfully!")
                print(f"Original file: {input_filename}")
                print(f"Converted file: {output_filename}")
            else:
                print("\nConversion failed. Please check the error messages above.")
        else:
            print("Conversion cancelled.")
    else:
        print("No patterns found to convert or error occurred during analysis.")