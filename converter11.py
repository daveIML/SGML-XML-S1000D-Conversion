# SGM_to_S1000D_Converter.py
import os
import re
from datetime import datetime
from lxml import etree

# --- Section 1: SGM Parser ---
# This section is responsible for parsing the SGML file.
# It uses the lxml HTML parser, which is robust enough to handle SGML's structure.

def parse_sgm(sgm_filepath):
    """
    Parses the SGM file using lxml's HTML parser.
    """
    try:
        with open(sgm_filepath, 'r', encoding='latin-1') as f:
            content = f.read()
        parser = etree.HTMLParser(recover=True, remove_comments=True)
        return etree.fromstring(f"<root>{content}</root>", parser)
    except Exception as e:
        print(f"Error parsing SGM file: {e}")
        return None

# --- Section 2: SGM-XML Element Map ---
# This section defines the mapping between SGML and S1000D elements.
# It includes helper functions and classes that represent the S1000D structure.

# Helper functions for text extraction and cleaning
def clean_text(text_node):
    """Cleans whitespace from a text node (string)."""
    if not text_node:
        return ""
    return re.sub(r'\s+', ' ', text_node).strip()

def get_full_text(element):
    """Extracts all text within an element, including children, and cleans it."""
    if element is None:
        return ""
    text = etree.tostring(element, method="text", encoding="unicode")
    return re.sub(r'\s+', ' ', text).strip()

# Base class for S1000D modules
class S1000D_Module:
    """Base class for S1000D PM and DM to handle common XML structure."""
    def __init__(self, output_dir, model_ident_code):
        self.output_dir = output_dir
        self.model_ident_code = model_ident_code
        self.root = None
        self.ident_status_section = None
        self.content = None
        self.issue_date = datetime.now().strftime("%Y-%m-%d")
        self.change_attrs = {}
        self.security_attrs = {'securityClassification': '01'}
        self.authority_attrs = {'authorityName': 'MIL-M-38784'}

    def _create_base_structure(self, root_tag):
        self.root = etree.Element(root_tag)
        self.ident_status_section = etree.SubElement(self.root, 'identAndStatusSection')
        self.content = etree.SubElement(self.root, 'content')

    def _add_status_section_common_items(self, status_element, title):
        etree.SubElement(status_element, 'security', securityClassification='01')
        etree.SubElement(status_element, 'responsiblePartnerCompany', enterpriseCode="00000").text = "GENERATED"
        etree.SubElement(status_element, 'originator', enterpriseCode="00000").text = "CONVERTER"
        
        qa = etree.SubElement(status_element, 'qualityAssurance')
        etree.SubElement(qa, 'unverified')
        brex_ref = etree.SubElement(status_element, 'brexDmRef')
        dm_ref = etree.SubElement(brex_ref, 'dmRef')
        dm_ref_ident = etree.SubElement(dm_ref, 'dmRefIdent')
        etree.SubElement(dm_ref_ident, 'dmCode', 
                         modelIdentCode="S1000D", systemDiffCode="A", systemCode="00",
                         subSystemCode="0", subSubSystemCode="0", assyCode="00",
                         disassyCode="00", disassyCodeVariant="A", infoCode="022",
                         infoCodeVariant="A", itemLocationCode="-")

    def to_string(self):
        return etree.tostring(self.root, pretty_print=True, xml_declaration=True, encoding="UTF-8")

    def write_to_file(self, filename):
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(self.to_string())
        print(f"  -> Generated file: {filepath}")

# Class for S1000D Publication Module
class PublicationModule(S1000D_Module):
    """Handles the creation of the S1000D Publication Module (PM)."""
    def __init__(self, output_dir, model_ident_code, title):
        super().__init__(output_dir, model_ident_code)
        self.pm_code_attrs = {'modelIdentCode': self.model_ident_code, 'pmIssuer': '00001', 'pmNumber': '00001', 'pmVolume': '01'}
        self.filename = f"PM-{'-'.join(self.pm_code_attrs.values())}.XML"
        self._create_base_structure('pm')
        self._build_ident_section(title)
        
    def _build_ident_section(self, title):
        pm_address = etree.SubElement(self.ident_status_section, 'pmAddress')
        pm_ident = etree.SubElement(pm_address, 'pmIdent')
        etree.SubElement(pm_ident, 'pmCode', **self.pm_code_attrs)
        etree.SubElement(pm_ident, 'language', languageIsoCode='en', countryIsoCode='US')
        etree.SubElement(pm_ident, 'issueInfo', issueNumber='001', inWork='00')
        pm_address_items = etree.SubElement(pm_address, 'pmAddressItems')
        etree.SubElement(pm_address_items, 'issueDate', year=self.issue_date[:4], month=self.issue_date[5:7], day=self.issue_date[8:10])
        etree.SubElement(pm_address_items, 'pmTitle').text = title
        pm_status = etree.SubElement(self.ident_status_section, 'pmStatus')
        self._add_status_section_common_items(pm_status, title)
        
    def add_dm_ref(self, parent_pm_entry, dm):
        dm_ref = etree.SubElement(parent_pm_entry, 'dmRef')
        dm_ref_ident = etree.SubElement(dm_ref, 'dmRefIdent')
        etree.SubElement(dm_ref_ident, 'dmCode', **dm.dm_code_attrs)
        dm_ref_addr = etree.SubElement(dm_ref, 'dmRefAddressItems')
        dm_title = etree.SubElement(dm_ref_addr, 'dmTitle')
        etree.SubElement(dm_title, 'techName').text = dm.tech_name
        etree.SubElement(dm_title, 'infoName').text = dm.info_name

# Class for S1000D Data Module
class DataModule(S1000D_Module):
    """Handles the creation of an S1000D Data Module (DM)."""
    def __init__(self, output_dir, model_ident_code, system_code, info_code, title):
        super().__init__(output_dir, model_ident_code)
        self.tech_name = title
        self.info_name = "Procedure" if info_code.startswith("2") else "Description"
        self.dm_code_attrs = {
            'modelIdentCode': self.model_ident_code, 'systemDiffCode': 'A',
            'systemCode': system_code, 'subSystemCode': '0', 'subSubSystemCode': '0',
            'assyCode': '00', 'disassyCode': '00', 'disassyCodeVariant': 'A',
            'infoCode': info_code, 'infoCodeVariant': 'A', 'itemLocationCode': '-'
        }
        self.filename = f"DMC-{'-'.join(self.dm_code_attrs.values())}.XML"
        self._create_base_structure('dmodule')
        self._build_ident_section(title)

    def _build_ident_section(self, title):
        dm_address = etree.SubElement(self.ident_status_section, 'dmAddress')
        dm_ident = etree.SubElement(dm_address, 'dmIdent')
        etree.SubElement(dm_ident, 'dmCode', **self.dm_code_attrs)
        etree.SubElement(dm_ident, 'language', languageIsoCode='en', countryIsoCode='US')
        etree.SubElement(dm_ident, 'issueInfo', issueNumber='001', inWork='00')
        dm_address_items = etree.SubElement(dm_address, 'dmAddressItems')
        etree.SubElement(dm_address_items, 'issueDate', year=self.issue_date[:4], month=self.issue_date[5:7], day=self.issue_date[8:10])
        dm_title = etree.SubElement(dm_address_items, 'dmTitle')
        etree.SubElement(dm_title, 'techName').text = self.tech_name
        etree.SubElement(dm_title, 'infoName').text = self.info_name
        dm_status = etree.SubElement(self.ident_status_section, 'dmStatus')
        self._add_status_section_common_items(dm_status, title)

    def _populate_content(self, s1000d_parent, sgm_parent):
        self._process_mixed_content(s1000d_parent, sgm_parent, is_block_level=True)

    def _process_mixed_content(self, s1000d_parent, sgm_parent, is_block_level=False):
        if sgm_parent.text and not is_block_level:
            text = clean_text(sgm_parent.text)
            if text:
                s1000d_parent.text = (s1000d_parent.text or '') + text
        
        for child in sgm_parent:
            tag = child.tag.lower()
            last_s1000d_child = s1000d_parent[-1] if len(s1000d_parent) > 0 else None

            if is_block_level:
                if tag == 'subpara1' and 
                    subParaProc =  
                dm_creation_tags = {'foreword', 'glossary', 'safesum', 'chapter', 'section', 'subsection', 'para0'}
                if tag in dm_creation_tags:
                    continue
                if tag == 'title':
                    continue
                
                block_handlers = {
                    'para': self._handle_para, 'para0': self._handle_para, 'subpara1': self._handle_para,
                    'subpara2': self._handle_para, 'subpara3': self._handle_para, 'subpara4': self._handle_para,
                    'subpara5': self._handle_para, 'subpara6': self._handle_para, 'subpara7': self._handle_para,
                    'step1': self._handle_step, 'step2': self._handle_step, 'step3': self._handle_step,
                    'step4': self._handle_step, 'step5': self._handle_step, 'step6': self._handle_step,
                    'step7': self._handle_step, 'warning': self._handle_attention, 'caution': self._handle_attention,
                    'note': self._handle_attention, 'danger': self._handle_attention, 'randlist': self._handle_list,
                    'seqlist': self._handle_list, 'deflist': self._handle_list, 'figure': self._handle_figure_table,
                    'table': self._handle_figure_table
                }
                
                if tag in block_handlers:
                    block_handlers[tag](s1000d_parent, child)
                    continue

            inline_handlers = {
                'xref': self._handle_xref, 'emphasis': self._handle_emphasis,
                'brk': self._handle_brk, 'item': self._handle_item
            }

            if tag in inline_handlers:
                inline_handlers[tag](s1000d_parent, child)
            else:
                unhandled_text = get_full_text(child)
                if unhandled_text:
                    if last_s1000d_child is not None:
                        last_s1000d_child.tail = (last_s1000d_child.tail or '') + ' ' + unhandled_text
                    else:
                        s1000d_parent.text = (s1000d_parent.text or '') + ' ' + unhandled_text

            if child.tail:
                tail_text = clean_text(child.tail)
                if tail_text:
                    current_last_child = s1000d_parent[-1] if len(s1000d_parent) > 0 else None
                    if current_last_child is not None:
                        current_last_child.tail = (current_last_child.tail or '') + ' ' + tail_text
                    else:
                        s1000d_parent.text = (s1000d_parent.text or '') + ' ' + tail_text

    def _handle_para(self, s1000d_grandparent, sgm_para):
        if s1000d_grandparent.tag == 'description':
            target_parent = etree.SubElement(s1000d_grandparent, 'levelledPara', **self.change_attrs)
        else:
            target_parent = s1000d_grandparent
        para_elem = etree.SubElement(target_parent, 'para', **self.change_attrs)
        self._process_mixed_content(para_elem, sgm_para)

    def _handle_step(self, s1000d_parent, sgm_step):
        step_elem = etree.SubElement(s1000d_parent, 'proceduralStep', **self.change_attrs, **self.security_attrs)
        para_elem = etree.SubElement(step_elem, 'para')
        self._process_mixed_content(para_elem, sgm_step)
    
    def _handle_attention(self, s1000d_parent, sgm_attention):
        tag = sgm_attention.tag.lower()
        s1000d_tag = 'warning' if tag == 'danger' else tag
        s1000d_elem = etree.SubElement(s1000d_parent, s1000d_tag, **self.change_attrs)
        if tag == 'danger': s1000d_elem.set('vitalWarningFlag', '1')
        para_tag = 'warningAndCautionPara' if s1000d_tag in ['warning', 'caution'] else 'notePara'
        para = etree.SubElement(s1000d_elem, para_tag)
        self._process_mixed_content(para, sgm_attention)

    def _handle_list(self, s1000d_parent, sgm_list):
        tag = sgm_list.tag.lower()
        list_map = {'randlist': 'randomList', 'seqlist': 'sequentialList', 'deflist': 'definitionList'}
        list_elem = etree.SubElement(s1000d_parent, list_map.get(tag, 'randomList'), **self.change_attrs)
        if sgm_list.find('title') is not None:
             etree.SubElement(list_elem, 'title').text = get_full_text(sgm_list.find('title'))
        
        for item in sgm_list.findall('item'):
            self._handle_item(list_elem, item)

    def _handle_figure_table(self, s1000d_parent, sgm_fig_or_table):
        tag = sgm_fig_or_table.tag.lower()
        s1000d_elem = etree.SubElement(s1000d_parent, tag, **self.change_attrs)
        title_elem = sgm_fig_or_table.find('title')
        if title_elem is not None:
            etree.SubElement(s1000d_elem, 'title').text = get_full_text(title_elem)
        graphic = sgm_fig_or_table.find('graphic')
        if tag == 'figure' and graphic is not None:
            etree.SubElement(s1000d_elem, 'graphic').set('infoEntityIdent', graphic.get('boardno', 'UNKNOWN'))
        elif tag == 'table' and sgm_fig_or_table.find('tgroup') is not None:
            etree.SubElement(s1000d_elem, 'tgroup', cols=sgm_fig_or_table.find('tgroup').get('cols', '1'))
            
    def _handle_xref(self, s1000d_parent, sgm_xref):
        ref = etree.SubElement(s1000d_parent, 'internalRef', internalRefId=sgm_xref.get('xrefid', ''))
        ref.text = get_full_text(sgm_xref)

    def _handle_emphasis(self, s1000d_parent, sgm_emphasis):
        emph = etree.SubElement(s1000d_parent, 'emphasis')
        emph.text = get_full_text(sgm_emphasis)
    
    def _handle_brk(self, s1000d_parent, sgm_brk):
        etree.SubElement(s1000d_parent, 'break')

    def _handle_item(self, s1000d_parent, sgm_item):
        list_item = etree.SubElement(s1000d_parent, 'listItem')
        para = etree.SubElement(list_item, 'para')
        self._process_mixed_content(para, sgm_item)

# --- Section 3: SGM-XML Conversion Tool ---
# This section contains the main conversion tool, which uses the parser and the element map.

class S1000D_Converter:
    """Main class to convert a NAVSEA SGM file to S1000D XML."""
    def __init__(self, sgm_filepath, output_dir="output"):
        self.sgm_filepath = sgm_filepath
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.pm = None
        self.dms = []
        self.dm_counter = 1
        self.dm_creation_tags = {'foreword', 'glossary', 'safesum', 'chapter', 'section', 'subsection', 'para0'}

    def convert(self):
        """Main conversion orchestrator."""
        print(f"Starting conversion of: {self.sgm_filepath}")
        sgm_root = parse_sgm(self.sgm_filepath)
        if sgm_root is None: return

        doc = sgm_root.find('.//docnavseac2')
        if doc is None:
            print("CRITICAL ERROR: Could not find the root <docnavseac2> element. Conversion cannot proceed.")
            return

        title_element = doc.find('.//prtitle/nomen')
        doc_title = get_full_text(title_element) if title_element is not None else "Untitled Publication"
        
        modelno_element = doc.find('.//prtitle/modelno')
        model_ident_code = get_full_text(modelno_element) if modelno_element is not None else "DEFAULT"
        model_ident_code = re.sub(r'[^A-Z0-9]', '', model_ident_code.upper())[:14]

        self.pm = PublicationModule(self.output_dir, model_ident_code, doc_title)
        
        print(f"Searching for specified DM creation tags: {', '.join(self.dm_creation_tags)}")
        self._process_sgm_node(doc, self.pm.content, model_ident_code)
        
        print("\nWriting S1000D files...")
        self.pm.write_to_file(self.pm.filename)
        for dm in self.dms:
            dm.write_to_file(dm.filename)
        
        if not self.dms:
            print("\nWARNING: Conversion complete, but no Data Modules were generated.")
            print("This may mean no elements matching the creation rules were found.")
        else:
            print(f"\nConversion complete. Generated 1 Publication Module and {len(self.dms)} Data Modules.")

    def _process_sgm_node(self, sgm_node, parent_pm_entry, model_ident_code):
        for child_node in sgm_node:
            tag = child_node.tag.lower()
            
            if tag in self.dm_creation_tags:
                title_elem = child_node.find('title')
                if title_elem is not None:
                    title = get_full_text(title_elem)
                else:
                    title = tag.capitalize().replace("Para0", "Procedure")
                
                print(f"  - Found '{title}' ({tag}). Creating Data Module...")
                
                sys_code = str(self.dm_counter).zfill(3)
                self.dm_counter += 1

                is_procedural = is_procedural_by_title(title)
                info_code = "200" if is_procedural else "100"

                dm = DataModule(self.output_dir, model_ident_code, sys_code, info_code, title)
                
                if is_procedural:
                    proc_root = etree.SubElement(dm.content, 'procedure')
                    etree.SubElement(proc_root, 'preliminaryRqmts')
                    main_proc = etree.SubElement(proc_root, 'mainProcedure')
                    dm._populate_content(main_proc, child_node)
                    etree.SubElement(proc_root, 'closeRqmts')
                else:
                    desc_root = etree.SubElement(dm.content, 'description')
                    dm._populate_content(desc_root, child_node)
                
                self.dms.append(dm)
                
                new_pm_entry = etree.SubElement(parent_pm_entry, 'pmEntry', pmEntryType='pmt02')
                etree.SubElement(new_pm_entry, 'pmEntryTitle').text = title
                self.pm.add_dm_ref(new_pm_entry, dm)
                
                self._process_sgm_node(child_node, new_pm_entry, model_ident_code)
            
            else:
                self._process_sgm_node(child_node, parent_pm_entry, model_ident_code)

# --- Main Execution ---
if __name__ == "__main__":
    print("SGM to S1000D Conversion Tool")
    print("=" * 30)
    
    path_input = input("Please enter the path to the .sgm file: ")
    sgm_file_path = path_input.strip().strip("'\"")
    
    if not os.path.exists(sgm_file_path):
        print(f"\nError: File not found at '{sgm_file_path}'")
    else:
        output_directory = "s1000d_output"
        print(f"\nInput file found: '{sgm_file_path}'")
        print(f"Output will be saved to the '{output_directory}' directory.")
        
        try:
            converter = S1000D_Converter(sgm_file_path, output_dir=output_directory)
            converter.convert()
        except PermissionError:
            print("\nCRITICAL ERROR: Permission denied.")
            print(f"Could not create the output directory: '{output_directory}'")
            print("Please run this script from a location where you have write permissions.")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
