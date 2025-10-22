pip install lxml beautifulsoup4

import os
import re
import tkinter as tk
from tkinter import filedialog
from lxml import etree
from bs4 import BeautifulSoup
import sys

# --- 1. File Selection ---

def get_file_paths():
    """Uses Tkinter to ask the user for all required files."""
    print("Opening file dialogs. Please select your files...")

    # Hide the empty root window
    root = tk.Tk()
    root.withdraw()

    # 1. Get SGML DTD
    dtd_path = filedialog.askopenfilename(
        title="1. Select the SGML DTD file",
        filetypes=[("DTD files", "*.dtd"), ("All files", "*.*")]
    )
    if not dtd_path:
        print("Cancelled. Exiting.")
        sys.exit()

    # 2. Get S1000D XSDs (can select multiple)
    xsd_paths = filedialog.askopenfilenames(
        title="2. Select S1000D 4.0.1 XSD file(s) (e.g., dm.xsd, common.xsd)",
        filetypes=[("XSD files", "*.xsd"), ("All files", "*.*")]
    )
    if not xsd_paths:
        print("Cancelled. Exiting.")
        sys.exit()

    # 3. Get the input SGML file
    sgm_path = filedialog.askopenfilename(
        title="3. Select the input .sgm file to transform",
        filetypes=[("SGML files", "*.sgm"), ("All files", "*.*")]
    )
    if not sgm_path:
        print("Cancelled. Exiting.")
        sys.exit()

    # 4. Get the output XML save location
    output_path = filedialog.asksaveasfilename(
        title="4. Choose a save location for the output .xml file",
        defaultextension=".xml",
        filetypes=[("XML files", "*.xml"), ("All files", "*.*")]
    )
    if not output_path:
        print("Cancelled. Exiting.")
        sys.exit()

    print("Files selected successfully.\n")
    return dtd_path, xsd_paths, sgm_path, output_path

# --- 2. Schema Parsing ---

def parse_dtd_elements(dtd_path):
    """
    Parses a DTD file with regex to find all element definitions.
    This is a simple heuristic but avoids needing a full DTD parser.
    """
    elements = set()
    # Regex to find '<!ELEMENT element-name ...'
    element_regex = re.compile(r'<!ELEMENT\s+([\w.-]+)', re.IGNORECASE)

    try:
        with open(dtd_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            for match in element_regex.finditer(content):
                elements.add(match.group(1).lower())
        print(f"Found {len(elements)} elements in DTD.")
        return elements
    except Exception as e:
        print(f"Error parsing DTD {dtd_path}: {e}")
        return set()

def parse_xsd_elements(xsd_paths):
    """Parses one or more XSD files to find all element definitions."""
    elements = set()
    # Namespace for XSD files
    ns = {'xsd': 'http://www.w3.org/2001/XMLSchema'}

    for path in xsd_paths:
        try:
            tree = etree.parse(path)
            # XPath to find all <xsd:element> tags that have a 'name' attribute
            nodes = tree.xpath('//xsd:element[@name]', namespaces=ns)
            for node in nodes:
                elements.add(node.get('name').lower())
        except Exception as e:
            print(f"Error parsing XSD {path}: {e}")

    print(f"Found {len(elements)} elements in S1000D XSD(s).")
    return elements

# --- 3. Interactive Mapping ---

def build_interactive_mapping(dtd_elements, xsd_elements):
    """
    Interactively prompts the user to map DTD elements to XSD elements.
    """
    print("\n--- Starting Interactive Element Mapping ---")
    print("For each SGML element, enter the S1000D element it maps to.")
    print("Special commands:")
    print("  DELETE - Remove this element and all its children.")
    print("  UNWRAP - Remove this element but keep its children.")
    print("  (just press Enter) - Keep the original element name (if it matches).\n")

    mapping = {}

    # Sort for a consistent mapping order
    for sgml_elem in sorted(list(dtd_elements)):

        # Check for a direct match (case-insensitive)
        if sgml_elem.lower() in xsd_elements:
            print(f"Auto-mapping: <{sgml_elem}> -> <{sgml_elem.lower()}>")
            mapping[sgml_elem.lower()] = sgml_elem.lower()
            continue

        # No direct match, so we must ask the user
        print("-------------------------------------------------")
        print(f"❓ How should SGML element <{sgml_elem}> be mapped?")

        user_input = input(f"   Map <{sgml_elem}> to: ").strip().lower()

        if not user_input:
            # User just pressed Enter, assume keep-as-is
            print(f"  -> Keeping original: <{sgml_elem}>")
            mapping[sgml_elem.lower()] = sgml_elem
        else:
            # Store the user's decision (e.g., 'para', 'delete', 'unwrap')
            mapping[sgml_elem.lower()] = user_input
            if user_input not in ['delete', 'unwrap'] and user_input not in xsd_elements:
                print(f"  ⚠️ Warning: '{user_input}' is not a recognized S1000D element.")
            print(f"  -> Mapping set: <{sgml_elem}> -> <{user_input}>")

    print("\n--- Element Mapping Complete ---")
    return mapping

# --- 4. SGM Transformation ---

def transform_sgm(sgm_path, output_path, mapping, xsd_paths):
    """
    Parses the SGM file and applies the mapping to transform it into XML.
    """
    print(f"\nTransforming {sgm_path}...")

    try:
        with open(sgm_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Use 'lxml-xml' parser for best-effort SGML/XML parsing
            soup = BeautifulSoup(f, 'lxml-xml')

        # Walk all tags in the document
        for tag in soup.find_all(True):
            tag_name_lower = tag.name.lower()

            # Look up the mapping for this tag
            target_action = mapping.get(tag_name_lower)

            if target_action:
                if target_action == "delete":
                    # Remove the tag and all its contents
                    tag.decompose()
                elif target_action == "unwrap":
                    # Remove the tag but keep its contents
                    tag.unwrap()
                else:
                    # Rename the tag
                    tag.name = target_action

        # Find the new root element to add S1000D namespaces
        root_element = None
        for child in soup.children:
            if isinstance(child, (etree._Element, etree._Comment, BeautifulSoup.Tag)):
                root_element = child
                break

        if root_element:
            print(f"Setting namespaces on root element: <{root_element.name}>")
            root_element['xmlns'] = "http://www.s1000d.org/S1000D_4-0-1"
            root_element['xmlns:xsi'] = "http://www.w3.org/2001/XMLSchema-instance"

            # Create the schemaLocation string. Assumes main XSD is first.
            main_xsd_name = os.path.basename(xsd_paths[0])
            root_element['xsi:schemaLocation'] = f"http://www.s1000d.org/S1000D_4-0-1 {main_xsd_name}"

        # Write the transformed XML to the output file
        with open(output_path, 'wb') as f:
            # Prettify and encode as UTF-8
            f.write(soup.prettify(encoding='utf-8'))

        print(f"\n✅ Success! Transformed file saved to:\n{output_path}")

    except Exception as e:
        print(f"\n❌ Error during transformation: {e}")
        print("This often happens if the SGML is too complex for the parser.")
        print("See 'Important Limitations' in the documentation.")

# --- 5. Main Execution ---

def main():
    print("--- SGM to S1000D 4.0.1 Mapper ---")

    # 1. Get files
    try:
        dtd_path, xsd_paths, sgm_path, output_path = get_file_paths()
    except Exception as e:
        print(f"Could not initialize file dialogs. Error: {e}")
        print("Please ensure you have a graphical environment (like Windows, macOS, or a Linux desktop) to run this script.")
        return

    # 2. Parse schemas
    dtd_elements = parse_dtd_elements(dtd_path)
    xsd_elements = parse_xsd_elements(xsd_paths)

    if not dtd_elements or not xsd_elements:
        print("Error: Could not parse DTD or XSD files. Exiting.")
        return

    # 3. Build mapping
    mapping = build_interactive_mapping(dtd_elements, xsd_elements)

    # 4. Transform
    transform_sgm(sgm_path, output_path, mapping, xsd_paths)

if __name__ == "__main__":
    main()
