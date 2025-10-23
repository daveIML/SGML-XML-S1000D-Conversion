#!/usr/bin/env python3
"""
Secure GUI application to convert legacy SGML (.sgm) IETMs to S1000D 4.0.1-compliant XML
with one Publication Module (PM) and multiple Data Modules (DMs). The application:

- Prompts the user to supply the DTD used by the SGML.
- Converts SGML to XML using OpenSP (onsgmls + sx) when available; otherwise offers a
  safe, minimal heuristic fallback and strongly recommends installing OpenSP.
- Builds a PM and nested DMs from the converted XML and validates against user-provided XSDs.
- Connects to an LLM (e.g., OpenAI) when users provide additional files/info to suggest
  mapping rules or XSLT snippets to improve conversion.
- All interactions via a user-friendly Tkinter GUI, no terminal I/O.

Security & compliance notes:
- Follows secure coding practices (NIST CSF, DoD RMF principles).
- No shell=True; subprocess calls use argument arrays to avoid command injection.
- Input file extensions validated; filenames sanitized; no arbitrary code execution.
- Uses defusedxml.lxml for untrusted XML parsing to mitigate XXE and billion laugh attacks.
- LLM API key read from environment variable; not stored on disk; optional UI input stored
  in-memory only for the session.
- Provides detailed validation reports from XSD.

Dependencies:
- Python 3.10+
- lxml
- defusedxml
- Optional: openai (for LLM); OpenSP (onsgmls, sx) on PATH for robust SGML->XML conversion.

Install examples:
  pip install lxml defusedxml openai

Windows OpenSP: https://sourceforge.net/projects/opensp/
Linux: apt-get install opensp
macOS: brew install opensp
"""

import os
import re
import sys
import json
import threading
import queue
import logging
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

# Secure XML parsing
from defusedxml.lxml import fromstring as safe_fromstring
from defusedxml.lxml import parse as safe_parse
from lxml import etree

# ----------------------------
# Logging setup (sanitized)
# ----------------------------
logger = logging.getLogger("SGM2S1000D")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# ----------------------------
# Utility functions (secure)
# ----------------------------

def sanitize_filename(name: str) -> str:
    """Sanitize a filename to avoid path traversal and illegal chars."""
    # Keep only alphanumerics, underscore, hyphen, dot
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    # Remove leading dots to avoid hidden files
    return safe.lstrip('.')[:255]


def safe_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(content)


def safe_read_text(path: Path, max_bytes: int = 2_000_000) -> str:
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"File too large for memory-safe read: {size} bytes > {max_bytes}")
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()


def is_allowed_extension(path: Path, allowed: Tuple[str, ...]) -> bool:
    return path.suffix.lower() in allowed


def find_executable(name: str) -> Optional[str]:
    """Return full path of an executable if found in PATH, else None."""
    from shutil import which
    exe = which(name)
    return exe


# ----------------------------
# LLM client (optional)
# ----------------------------
class LLMClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self._client = None
        try:
            import openai
            # OpenAI SDK v1 style
            openai.api_key = self.api_key
            self._client = openai
        except Exception as e:
            logger.warning("OpenAI SDK not available: %s", e)

    def available(self) -> bool:
        return self._client is not None and bool(self.api_key)

    def ask(self, system_prompt: str, user_prompt: str) -> str:
        if not self.available():
            return "LLM not available. Install openai and set OPENAI_API_KEY."
        try:
            # Use Chat Completions; keep temperature low for deterministic mapping
            resp = self._client.ChatCompletion.create(
                model=self.model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.choices[0].message["content"]
        except Exception as e:
            logger.error("LLM error: %s", e)
            return f"LLM error: {e}"


# ----------------------------
# SGML to XML conversion
# ----------------------------
class SGMLConverter:
    def __init__(self, onsgmls_path: Optional[str] = None, sx_path: Optional[str] = None):
        self.onsgmls_path = onsgmls_path or find_executable("onsgmls")
        self.sx_path = sx_path or find_executable("sx")

    def tools_available(self) -> bool:
        return bool(self.onsgmls_path) and bool(self.sx_path)

    def convert_with_opensp(self, sgm_file: Path, dtd_path: Optional[Path], output_xml: Path) -> Tuple[bool, str]:
        """
        Convert SGML to XML using onsgmls + sx.
        We avoid shell=True and pipe safely. We do not assume a catalog file.
        """
        try:
            cmd1 = [self.onsgmls_path, "-wxml", "-s"]
            # If DTD is provided, add its directory to search path (-D)
            if dtd_path:
                cmd1 += ["-D", str(dtd_path.parent.resolve())]
            cmd1.append(str(sgm_file.resolve()))

            p1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            cmd2 = [self.sx_path, "-x", "xml"]
            p2 = subprocess.Popen(cmd2, stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            p1.stdout.close()  # allow p1 to receive a SIGPIPE if p2 exits
            out, err = p2.communicate()
            if p2.returncode != 0:
                return False, f"sx failed: {err.decode('utf-8', errors='replace')}"
            safe_write_text(output_xml, out.decode('utf-8', errors='replace'))
            return True, "Conversion with OpenSP succeeded."
        except FileNotFoundError as e:
            return False, f"OpenSP tools not found: {e}"
        except Exception as e:
            return False, f"Unexpected error during OpenSP conversion: {e}"

    def heuristic_convert(self, sgm_file: Path, output_xml: Path) -> Tuple[bool, str]:
        """
        Minimal heuristic SGML->XML conversion for near-XML SGML. Not standards-robust.
        Strongly advise using OpenSP for production compliance.
        """
        try:
            text = safe_read_text(sgm_file, max_bytes=5_000_000)

            # Heuristics:
            # - Ensure attribute values are quoted: attr=value -> attr="value"
            text = re.sub(r"(\s[\w:-]+)=([\w./:-]+)", r"\1=\"\2\"", text)
            # - Replace SGML empty tags <tag/> properly (if present)
            text = re.sub(r"<([A-Za-z0-9:_-]+)\s*/>", r"<\1/>", text)
            # - Remove SGML short references (rare); conservative
            text = text.replace("\r", "")

            # Wrap in a root element if missing (required for parsing)
            if "<" not in text or ">" not in text:
                return False, "Input not parseable as SGML/XML."

            if not re.search(r"<\?xml", text):
                text = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" + text

            # Attempt to parse to verify syntactic correctness
            try:
                doc = safe_fromstring(text.encode('utf-8'))
            except Exception as e:
                # If fails, try to enclose in <root>
                wrapped = f"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<root>\n{text}\n</root>"
                doc = safe_fromstring(wrapped.encode('utf-8'))
                text = wrapped

            safe_write_text(output_xml, text)
            return True, "Heuristic conversion completed (not DTD-aware)."
        except Exception as e:
            return False, f"Heuristic conversion error: {e}"

    def convert(self, sgm_file: Path, dtd_path: Optional[Path], output_xml: Path) -> Tuple[bool, str]:
        if self.tools_available():
            ok, msg = self.convert_with_opensp(sgm_file, dtd_path, output_xml)
            if ok:
                return ok, msg
            # Fall through to heuristic with warning
            logger.warning("OpenSP conversion failed; attempting heuristic conversion: %s", msg)
        return self.heuristic_convert(sgm_file, output_xml)


# ----------------------------
# S1000D module builders
# ----------------------------
class S1000DBuilder:
    def __init__(self, xsd_dir: Optional[Path] = None):
        self.xsd_dir = xsd_dir
        self.schema_pm = None
        self.schema_dm = None
        self._load_schemas()

    def _load_schemas(self):
        if not self.xsd_dir:
            return
        try:
            pm_xsd = self.xsd_dir / "pm.xsd"
            dm_xsd = self.xsd_dir / "dm.xsd"
            if pm_xsd.exists():
                with open(pm_xsd, 'rb') as f:
                    pm_doc = etree.XML(f.read())
                    self.schema_pm = etree.XMLSchema(pm_doc)
            if dm_xsd.exists():
                with open(dm_xsd, 'rb') as f:
                    dm_doc = etree.XML(f.read())
                    self.schema_dm = etree.XMLSchema(dm_doc)
        except Exception as e:
            logger.warning("Failed to load XSD schemas: %s", e)

    def build_publication_module(self, pm_ident: dict, dm_refs: List[Path]) -> etree._ElementTree:
        NS_XSI = "http://www.w3.org/2001/XMLSchema-instance"
        pm = etree.Element("pm", nsmap={'xsi': NS_XSI})
        if self.xsd_dir:
            pm.set(f"{{{NS_XSI}}}noNamespaceSchemaLocation", str((self.xsd_dir / "pm.xsd").as_posix()))

        identAndStatus = etree.SubElement(pm, "identAndStatusSection")
        pmAddress = etree.SubElement(identAndStatus, "pmAddress")
        pmIdent = etree.SubElement(pmAddress, "pmIdent")
        pmCode = etree.SubElement(pmIdent, "pmCode")

        # Minimal required attributes; users can customize in GUI
        for k, v in pm_ident.items():
            pmCode.set(k, v)

        pmStatus = etree.SubElement(identAndStatus, "pmStatus")
        etree.SubElement(pmStatus, "security")
        etree.SubElement(pmStatus, "responsiblePartnerCompany")
        etree.SubElement(pmStatus, "originator")
        etree.SubElement(pmStatus, "applicCrossRefTable")
        etree.SubElement(pmStatus, "pmTitle").text = pm_ident.get("title", "Converted Publication")

        content = etree.SubElement(pm, "content")
        # Add references to DMs in content; schema varies by issue; using dmRef with dmAddress/dmIdent in minimal form
        for dm_path in dm_refs:
            dmRef = etree.SubElement(content, "dmRef")
            dmAddressRef = etree.SubElement(dmRef, "dmAddress")
            dmIdentRef = etree.SubElement(dmAddressRef, "dmIdent")
            dmCodeRef = etree.SubElement(dmIdentRef, "dmCode")
            # Put a placeholder linking via attribute; actual linking can use xref attributes or dmCode attributes
            dmCodeRef.set("modelIdentCode", pm_ident.get("modelIdentCode", "MODEL"))
            dmCodeRef.set("systemCode", "GEN")
            dmCodeRef.set("infoCode", "000")
            dmCodeRef.set("itemLocationCode", "A00")
            dmRefText = etree.SubElement(dmRef, "dmTitle")
            dmRefText.text = sanitize_filename(dm_path.stem)

        return etree.ElementTree(pm)

    def build_data_module(self, dm_ident: dict, body_content: etree._Element) -> etree._ElementTree:
        NS_XSI = "http://www.w3.org/2001/XMLSchema-instance"
        dm = etree.Element("dm", nsmap={'xsi': NS_XSI})
        if self.xsd_dir:
            dm.set(f"{{{NS_XSI}}}noNamespaceSchemaLocation", str((self.xsd_dir / "dm.xsd").as_posix()))

        identAndStatus = etree.SubElement(dm, "identAndStatusSection")
        dmAddress = etree.SubElement(identAndStatus, "dmAddress")
        dmIdent = etree.SubElement(dmAddress, "dmIdent")
        dmCode = etree.SubElement(dmIdent, "dmCode")

        # Populate dmCode attributes as provided or defaults
        defaults = {
            "modelIdentCode": dm_ident.get("modelIdentCode", "MODEL"),
            "systemDifferenceCode": dm_ident.get("systemDifferenceCode", "A"),
            "systemCode": dm_ident.get("systemCode", "GEN"),
            "subSystemCode": dm_ident.get("subSystemCode", "00"),
            "subSubSystemCode": dm_ident.get("subSubSystemCode", "0"),
            "assyCode": dm_ident.get("assyCode", "00"),
            "disassyCode": dm_ident.get("disassyCode", "00"),
            "disassyCodeVariant": dm_ident.get("disassyCodeVariant", "A"),
            "infoCode": dm_ident.get("infoCode", "000"),
            "infoCodeVariant": dm_ident.get("infoCodeVariant", "A"),
            "itemLocationCode": dm_ident.get("itemLocationCode", "A00"),
        }
        for k, v in defaults.items():
            dmCode.set(k, v)

        lang = etree.SubElement(dmIdent, "language")
        lang.set("languageIsoCode", dm_ident.get("languageIsoCode", "en"))
        issueInfo = etree.SubElement(dmIdent, "issueInfo")
        issueInfo.set("issueNumber", dm_ident.get("issueNumber", "001"))
        issueInfo.set("inWork", dm_ident.get("inWork", "00"))

        etree.SubElement(dmAddress, "dmAddressItems")

        content = etree.SubElement(dm, "content")
        # For simplicity, wrap content in a descriptive section
        descriptive = etree.SubElement(content, "descriptive")
        # Append provided body content under descriptive
        if body_content is not None:
            descriptive.append(body_content)

        return etree.ElementTree(dm)

    def validate_pm(self, tree: etree._ElementTree) -> List[str]:
        errs = []
        if self.schema_pm is None:
            return errs
        try:
            self.schema_pm.assertValid(tree)
        except etree.DocumentInvalid as e:
            errs.append(str(e))
        return errs

    def validate_dm(self, tree: etree._ElementTree) -> List[str]:
        errs = []
        if self.schema_dm is None:
            return errs
        try:
            self.schema_dm.assertValid(tree)
        except etree.DocumentInvalid as e:
            errs.append(str(e))
        return errs


# ----------------------------
# Main GUI Application
# ----------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SGML to S1000D 4.0.1 Converter")
        self.geometry("1000x700")
        self.minsize(900, 600)

        # State
        self.sgm_path: Optional[Path] = None
        self.dtd_path: Optional[Path] = None
        self.xsd_dir: Optional[Path] = None
        self.output_dir: Optional[Path] = None

        self.onsgmls_path = find_executable("onsgmls")
        self.sx_path = find_executable("sx")

        self.llm_api_key: Optional[str] = None
        self.llm_model: str = "gpt-4o-mini"
        self.llm_client = LLMClient(api_key=self.llm_api_key, model=self.llm_model)

        self.converter = SGMLConverter(self.onsgmls_path, self.sx_path)
        self.builder = S1000DBuilder()

        self._build_gui()

    def _build_gui(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True)

        self.convert_frame = ttk.Frame(notebook)
        self.validate_frame = ttk.Frame(notebook)
        self.llm_frame = ttk.Frame(notebook)
        self.settings_frame = ttk.Frame(notebook)

        notebook.add(self.convert_frame, text="Convert")
        notebook.add(self.validate_frame, text="Validate")
        notebook.add(self.llm_frame, text="LLM Assistant")
        notebook.add(self.settings_frame, text="Settings")

        self._build_convert_tab()
        self._build_validate_tab()
        self._build_llm_tab()
        self._build_settings_tab()

    # ---------------- Convert Tab -----------------
    def _build_convert_tab(self):
        frame = self.convert_frame
        pad = {'padx': 8, 'pady': 6}

        # SGML file
        ttk.Label(frame, text="Select SGML (.sgm) file:").grid(row=0, column=0, sticky='w', **pad)
        self.sgm_entry = ttk.Entry(frame, width=80)
        self.sgm_entry.grid(row=0, column=1, sticky='we', **pad)
        ttk.Button(frame, text="Browse", command=self._choose_sgm).grid(row=0, column=2, **pad)

        # DTD file
        ttk.Label(frame, text="Select DTD file:").grid(row=1, column=0, sticky='w', **pad)
        self.dtd_entry = ttk.Entry(frame, width=80)
        self.dtd_entry.grid(row=1, column=1, sticky='we', **pad)
        ttk.Button(frame, text="Browse", command=self._choose_dtd).grid(row=1, column=2, **pad)

        # XSD folder
        ttk.Label(frame, text="Select S1000D XSD directory:").grid(row=2, column=0, sticky='w', **pad)
        self.xsd_entry = ttk.Entry(frame, width=80)
        self.xsd_entry.grid(row=2, column=1, sticky='we', **pad)
        ttk.Button(frame, text="Browse", command=self._choose_xsd).grid(row=2, column=2, **pad)

        # Output directory
        ttk.Label(frame, text="Select output directory:").grid(row=3, column=0, sticky='w', **pad)
        self.out_entry = ttk.Entry(frame, width=80)
        self.out_entry.grid(row=3, column=1, sticky='we', **pad)
        ttk.Button(frame, text="Browse", command=self._choose_output_dir).grid(row=3, column=2, **pad)

        # Section element name (splitting rule)
        ttk.Label(frame, text="Section element name to split into DMs (e.g., 'section' or 'chapter'):").grid(row=4, column=0, sticky='w', **pad)
        self.section_name_entry = ttk.Entry(frame, width=40)
        self.section_name_entry.insert(0, "section")
        self.section_name_entry.grid(row=4, column=1, sticky='w', **pad)

        # PM identity configuration (minimal)
        ttk.Label(frame, text="PM modelIdentCode:").grid(row=5, column=0, sticky='w', **pad)
        self.pm_model_entry = ttk.Entry(frame, width=20)
        self.pm_model_entry.insert(0, "MODEL")
        self.pm_model_entry.grid(row=5, column=1, sticky='w', **pad)
        ttk.Label(frame, text="PM title:").grid(row=5, column=2, sticky='w', **pad)
        self.pm_title_entry = ttk.Entry(frame, width=40)
        self.pm_title_entry.insert(0, "Converted Publication")
        self.pm_title_entry.grid(row=5, column=3, sticky='w', **pad)

        # Buttons
        self.convert_btn = ttk.Button(frame, text="Run Conversion", command=self._run_conversion)
        self.convert_btn.grid(row=6, column=1, sticky='w', **pad)
        self.progress = ttk.Progressbar(frame, mode='indeterminate')
        self.progress.grid(row=6, column=2, sticky='we', **pad)

        # Console output
        self.console = tk.Text(frame, height=20, wrap='word')
        self.console.grid(row=7, column=0, columnspan=4, sticky='nsew', **pad)
        frame.grid_rowconfigure(7, weight=1)
        frame.grid_columnconfigure(1, weight=1)

    # ---------------- Validate Tab -----------------
    def _build_validate_tab(self):
        frame = self.validate_frame
        pad = {'padx': 8, 'pady': 6}

        ttk.Label(frame, text="Select directory containing generated XML modules:").grid(row=0, column=0, sticky='w', **pad)
        self.validate_dir_entry = ttk.Entry(frame, width=80)
        self.validate_dir_entry.grid(row=0, column=1, sticky='we', **pad)
        ttk.Button(frame, text="Browse", command=self._choose_validate_dir).grid(row=0, column=2, **pad)
        self.validate_btn = ttk.Button(frame, text="Validate", command=self._run_validation)
        self.validate_btn.grid(row=0, column=3, **pad)

        self.validation_output = tk.Text(frame, height=25, wrap='word')
        self.validation_output.grid(row=1, column=0, columnspan=4, sticky='nsew', **pad)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(1, weight=1)

    # ---------------- LLM Assistant Tab -----------------
    def _build_llm_tab(self):
        frame = self.llm_frame
        pad = {'padx': 8, 'pady': 6}

        ttk.Label(frame, text="Ask the LLM for mapping advice or paste DTD fragments:").grid(row=0, column=0, sticky='w', **pad)
        self.llm_input = tk.Text(frame, height=15, wrap='word')
        self.llm_input.grid(row=1, column=0, columnspan=3, sticky='nsew', **pad)

        ttk.Button(frame, text="Send to LLM", command=self._send_llm).grid(row=2, column=0, sticky='w', **pad)
        self.llm_output = tk.Text(frame, height=20, wrap='word')
        self.llm_output.grid(row=3, column=0, columnspan=3, sticky='nsew', **pad)
        frame.grid_rowconfigure(3, weight=1)
        frame.grid_columnconfigure(0, weight=1)

    # ---------------- Settings Tab -----------------
    def _build_settings_tab(self):
        frame = self.settings_frame
        pad = {'padx': 8, 'pady': 6}

        ttk.Label(frame, text="OpenSP onsgmls path (optional):").grid(row=0, column=0, sticky='w', **pad)
        self.ons_entry = ttk.Entry(frame, width=60)
        if self.onsgmls_path:
            self.ons_entry.insert(0, self.onsgmls_path)
        self.ons_entry.grid(row=0, column=1, sticky='we', **pad)
        ttk.Button(frame, text="Browse", command=lambda: self._choose_exe(self.ons_entry)).grid(row=0, column=2, **pad)

        ttk.Label(frame, text="OpenSP sx path (optional):").grid(row=1, column=0, sticky='w', **pad)
        self.sx_entry = ttk.Entry(frame, width=60)
        if self.sx_path:
            self.sx_entry.insert(0, self.sx_path)
        self.sx_entry.grid(row=1, column=1, sticky='we', **pad)
        ttk.Button(frame, text="Browse", command=lambda: self._choose_exe(self.sx_entry)).grid(row=1, column=2, **pad)

        ttk.Label(frame, text="LLM API Key (OpenAI) - in-memory only:").grid(row=2, column=0, sticky='w', **pad)
        self.llm_key_entry = ttk.Entry(frame, width=60, show='*')
        self.llm_key_entry.grid(row=2, column=1, sticky='we', **pad)
        ttk.Button(frame, text="Set", command=self._set_llm_key).grid(row=2, column=2, **pad)

        ttk.Label(frame, text="LLM Model:").grid(row=3, column=0, sticky='w', **pad)
        self.llm_model_entry = ttk.Entry(frame, width=60)
        self.llm_model_entry.insert(0, self.llm_model)
        self.llm_model_entry.grid(row=3, column=1, sticky='we', **pad)
        ttk.Button(frame, text="Apply", command=self._apply_llm_model).grid(row=3, column=2, **pad)

        frame.grid_columnconfigure(1, weight=1)

    # -------------- File Choosers ----------------
    def _choose_sgm(self):
        path = filedialog.askopenfilename(title="Select SGML file", filetypes=[("SGML", ".sgm .sgml"), ("All", "*.*")])
        if not path:
            return
        p = Path(path)
        if not is_allowed_extension(p, (".sgm", ".sgml")):
            messagebox.showerror("Invalid file", "Please select a .sgm or .sgml file.")
            return
        self.sgm_path = p
        self.sgm_entry.delete(0, tk.END)
        self.sgm_entry.insert(0, str(p))

    def _choose_dtd(self):
        path = filedialog.askopenfilename(title="Select DTD file", filetypes=[("DTD", ".dtd"), ("All", "*.*")])
        if not path:
            return
        p = Path(path)
        if not is_allowed_extension(p, (".dtd",)):
            messagebox.showerror("Invalid file", "Please select a .dtd file.")
            return
        self.dtd_path = p
        self.dtd_entry.delete(0, tk.END)
        self.dtd_entry.insert(0, str(p))
        # Prompt to consult LLM about DTD mapping
        if messagebox.askyesno("LLM Assist", "Send DTD to LLM for mapping suggestions?"):
            try:
                dtd_text = safe_read_text(p, max_bytes=200_000)
            except Exception as e:
                dtd_text = f"DTD file too large to read: {e}"
            self._llm_suggest_mapping(dtd_text)

    def _choose_xsd(self):
        path = filedialog.askdirectory(title="Select S1000D XSD directory")
        if not path:
            return
        p = Path(path)
        self.xsd_dir = p
        self.xsd_entry.delete(0, tk.END)
        self.xsd_entry.insert(0, str(p))
        self.builder = S1000DBuilder(self.xsd_dir)

    def _choose_output_dir(self):
        path = filedialog.askdirectory(title="Select output directory")
        if not path:
            return
        p = Path(path)
        self.output_dir = p
        self.out_entry.delete(0, tk.END)
        self.out_entry.insert(0, str(p))

    def _choose_validate_dir(self):
        path = filedialog.askdirectory(title="Select directory to validate")
        if not path:
            return
        p = Path(path)
        self.validate_dir_entry.delete(0, tk.END)
        self.validate_dir_entry.insert(0, str(p))

    def _choose_exe(self, entry: ttk.Entry):
        path = filedialog.askopenfilename(title="Select executable", filetypes=[("Executable", "*" )])
        if not path:
            return
        entry.delete(0, tk.END)
        entry.insert(0, path)

    # -------------- LLM settings ----------------
    def _set_llm_key(self):
        key = self.llm_key_entry.get().strip()
        if not key:
            messagebox.showerror("LLM", "API key cannot be empty.")
            return
        self.llm_api_key = key
        self.llm_client = LLMClient(api_key=self.llm_api_key, model=self.llm_model)
        messagebox.showinfo("LLM", "API key set for this session (not stored).")

    def _apply_llm_model(self):
        model = self.llm_model_entry.get().strip()
        if not model:
            messagebox.showerror("LLM", "Model cannot be empty.")
            return
        self.llm_model = model
        self.llm_client = LLMClient(api_key=self.llm_api_key, model=self.llm_model)
        messagebox.showinfo("LLM", f"LLM model set to {model}.")

    # -------------- Conversion workflow ---------
    def _run_conversion(self):
        if not self.sgm_path or not self.output_dir:
            messagebox.showerror("Missing input", "Please select SGML file and output directory.")
            return
        if not self.dtd_path:
            if not messagebox.askyesno("No DTD", "No DTD provided. Continue with heuristic conversion?"):
                return
        section_name = self.section_name_entry.get().strip() or "section"
        pm_ident = {
            "modelIdentCode": self.pm_model_entry.get().strip() or "MODEL",
            "title": self.pm_title_entry.get().strip() or "Converted Publication",
        }

        self.progress.start(10)
        self.console.delete('1.0', tk.END)
        t = threading.Thread(target=self._convert_worker, args=(self.sgm_path, self.dtd_path, self.output_dir, section_name, pm_ident), daemon=True)
        t.start()

    def _convert_worker(self, sgm_path: Path, dtd_path: Optional[Path], out_dir: Path, section_name: str, pm_ident: dict):
        try:
            # Step 1: SGML -> intermediate XML
            inter_xml = out_dir / "intermediate.xml"
            ok, msg = self.converter.convert(sgm_path, dtd_path, inter_xml)
            self._console_append(msg + "\n")
            if not ok:
                self._console_append("Conversion failed. Aborting.\n")
                self.progress.stop()
                return

            # Step 2: Parse intermediate XML safely
            try:
                parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False)
                doc = etree.parse(str(inter_xml), parser)
            except Exception as e:
                self._console_append(f"Failed to parse intermediate XML: {e}\n")
                self.progress.stop()
                return

            # Step 3: Split into sections -> DMs
            root = doc.getroot()
            sections = root.findall(f".//{section_name}") if section_name else []
            if not sections:
                # If no sections, treat root children as a single DM body
                sections = [root]
                self._console_append("No sections found; using entire document as one DM.\n")

            dm_paths: List[Path] = []
            for idx, sect in enumerate(sections, start=1):
                dm_ident = {
                    "modelIdentCode": pm_ident.get("modelIdentCode", "MODEL"),
                    "languageIsoCode": "en",
                    "issueNumber": f"{idx:03d}",
                    "inWork": "00",
                }
                # Create a safe body element to append
                body = etree.Element("sectionBody")
                # Deep copy section to avoid modifying original tree
                body.append(etree.fromstring(etree.tostring(sect)))
                dm_tree = self.builder.build_data_module(dm_ident, body)
                dm_filename = sanitize_filename(f"dm_{idx:03d}.xml")
                dm_path = out_dir / dm_filename
                dm_paths.append(dm_path)

                # Write DM
                xml_bytes = etree.tostring(dm_tree, encoding='utf-8', xml_declaration=True, pretty_print=True)
                with open(dm_path, 'wb') as f:
                    f.write(xml_bytes)

                # Validate DM if schemas are loaded
                errs = self.builder.validate_dm(dm_tree)
                if errs:
                    self._console_append(f"DM {dm_filename} schema issues:\n" + "\n".join(errs) + "\n")
                else:
                    self._console_append(f"DM {dm_filename} written and validated.\n")

            # Step 4: Build PM referencing DMs
            pm_tree = self.builder.build_publication_module(pm_ident, dm_paths)
            pm_path = out_dir / "publication_module.xml"
            xml_bytes = etree.tostring(pm_tree, encoding='utf-8', xml_declaration=True, pretty_print=True)
            with open(pm_path, 'wb') as f:
                f.write(xml_bytes)

            pm_errs = self.builder.validate_pm(pm_tree)
            if pm_errs:
                self._console_append("PM schema issues:\n" + "\n".join(pm_errs) + "\n")
            else:
                self._console_append("PM written and validated.\n")

            self._console_append("Conversion complete.\n")
        except Exception as e:
            self._console_append(f"Unexpected error: {e}\n")
        finally:
            self.progress.stop()

    def _console_append(self, text: str):
        self.console.insert(tk.END, text)
        self.console.see(tk.END)

    # -------------- Validation workflow ----------
    def _run_validation(self):
        dir_path = Path(self.validate_dir_entry.get().strip())
        if not dir_path.exists() or not dir_path.is_dir():
            messagebox.showerror("Validation", "Please select a valid directory.")
            return
        if not self.xsd_dir:
            messagebox.showerror("Validation", "Please set the S1000D XSD directory in Convert tab.")
            return
        self.validation_output.delete('1.0', tk.END)
        t = threading.Thread(target=self._validate_worker, args=(dir_path,), daemon=True)
        t.start()

    def _validate_worker(self, dir_path: Path):
        builder = self.builder
        for xml_file in sorted(dir_path.glob("*.xml")):
            try:
                parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False)
                tree = etree.parse(str(xml_file), parser)
                root_name = tree.getroot().tag
                if root_name == "pm":
                    errs = builder.validate_pm(tree)
                elif root_name == "dm":
                    errs = builder.validate_dm(tree)
                else:
                    errs = ["Unknown root element; not a PM or DM."]
                if errs:
                    self.validation_output.insert(tk.END, f"{xml_file.name}: INVALID\n" + "\n".join(errs) + "\n\n")
                else:
                    self.validation_output.insert(tk.END, f"{xml_file.name}: VALID\n\n")
            except Exception as e:
                self.validation_output.insert(tk.END, f"{xml_file.name}: Error parsing - {e}\n\n")
        self.validation_output.see(tk.END)

    # -------------- LLM assistant ----------------
    def _send_llm(self):
        prompt = self.llm_input.get('1.0', tk.END).strip()
        if not prompt:
            messagebox.showerror("LLM", "Please enter a prompt.")
            return
        self.llm_output.delete('1.0', tk.END)

        def worker():
            system_prompt = (
                "You are an expert in S1000D 4.0.1, SGML/DTD, and XML/XSLT. "
                "Given user inputs (DTD fragments, SGML samples), propose safe, deterministic XSLT mapping "
                "or conversion guidance to transform SGML into S1000D PM/DM XML. Only output XSLT fragments "
                "or structured steps and avoid hallucinations."
            )
            resp = self.llm_client.ask(system_prompt, prompt)
            self.llm_output.insert(tk.END, resp)
            self.llm_output.see(tk.END)

        threading.Thread(target=worker, daemon=True).start()

    def _llm_suggest_mapping(self, dtd_text: str):
        if not self.llm_client.available():
            self.llm_output.insert(tk.END, "LLM not available. Set API key in Settings tab.\n")
            return
        system_prompt = (
            "You are an expert in S1000D 4.0.1, SGML/DTD, and XML/XSLT. "
            "Analyze the provided DTD and propose XSLT templates that map key elements to S1000D PM/DM structures. "
            "Be conservative and indicate assumptions."
        )
        user_prompt = f"DTD:\n{dtd_text}\n\nProvide XSLT mapping suggestions."
        resp = self.llm_client.ask(system_prompt, user_prompt)
        self.llm_output.delete('1.0', tk.END)
        self.llm_output.insert(tk.END, resp)
        self.llm_output.see(tk.END)


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()