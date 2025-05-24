# pdf2md_fast.py (Streamlit GUI版)
# ------------------------------------------------------------
# 依存:
#   pip install "pymupdf<1.25" streamlit yomi-toku
#   # OCR フォールバック用 – CUDA or CPU で動作
#   # Poppler が不要な純 PyMuPDF ルート
# ------------------------------------------------------------
import os, sys, json, hashlib, shutil, subprocess, tempfile, concurrent.futures, time
from pathlib import Path
from typing import List, Dict, Callable

import fitz                # PyMuPDF
import streamlit as st
import tkinter as tk
from tkinter import filedialog

# --------- ユーティリティ (変更なし) ------------------------------------
def sha256(fp: Path) -> str:
    h = hashlib.sha256()
    with fp.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# --------- 1. テキスト層判定 & Markdown 生成 (変更なし) ------------------
_MIN_CHARS = 30

def has_text_layer(page: fitz.Page, min_chars: int = _MIN_CHARS) -> bool:
    return len(page.get_text()) >= min_chars

def page_to_md(page: fitz.Page) -> str:
    text_dict = page.get_text("dict")
    lines = []
    for block in text_dict["blocks"]:
        if block["type"] != 0: # type 0 is text block
            continue
        for line in block["lines"]:
            if not line["spans"]: 
                continue
            span = line["spans"][0]
            lines.append({"size": span["size"], "text": span["text"].rstrip()})
    if not lines:
        return ""
    
    sizes = sorted({l["size"] for l in lines}, reverse=True)
    h1_size, h2_size = sizes[0], sizes[1] if len(sizes) > 1 else sizes[0]
    
    md_lines = []
    for l in lines:
        txt = l["text"]
        if not txt:
            continue
        if l["size"] >= h1_size:
            md_lines.append(f"# {txt}")
        elif l["size"] >= h2_size:
            md_lines.append(f"## {txt}")
        elif txt.lstrip().startswith(("•", "・", "〇", "◯", "-", "―", "–", "*")):
            md_lines.append(f"- {txt.lstrip('•・〇◯-–―* ')}")
        else:
            md_lines.append(txt)
    return "\n".join(md_lines)

# --------- 2. OCR フォールバック (変更なし, 一時ディレクトリ処理改善済み) -------------
def export_pages_as_png(doc: fitz.Document,
                        indices: List[int],
                        dpi: int = 220,
                        outdir: Path = None) -> List[Path]:
    if outdir is None:
        try:
            outdir = Path(tempfile.mkdtemp(prefix="pdf2md_png_"))
        except Exception as e:
            st.error(f"PNGエクスポート用の一時ディレクトリ作成に失敗: {e}")
            return []
    else:
        outdir = Path(outdir)
    
    try:
        outdir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        st.error(f"PNGエクスポート先のディレクトリ作成/確認に失敗 ({outdir}): {e}")
        return []

    png_paths = []
    for i in indices:
        try:
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=dpi)
            p = outdir / f"page_{i+1}.png"
            pix.save(p)
            png_paths.append(p)
        except Exception as e:
            st.warning(f"ページ {i+1} のPNGエクスポートに失敗: {e}")
    return png_paths

def run_yomitoku(png_paths: List[Path],
                 device: str = "cuda") -> Dict[int, str]:
    if not png_paths:
        return {}
    
    md_by_page = {}
    try:
        with tempfile.TemporaryDirectory(prefix="yomitoku_ocr_out_") as ocr_output_tmpdir_str, \
             tempfile.TemporaryDirectory(prefix="yomitoku_input_img_") as input_img_tmpdir_str:
            
            ocr_output_tmpdir_path = Path(ocr_output_tmpdir_str)
            input_img_tmpdir_path = Path(input_img_tmpdir_str)
                
            copied_png_paths_for_yomitoku = []
            original_indices_map = {} 

            for png_path in png_paths:
                try:
                    original_page_index = int(png_path.stem.split('_')[1]) - 1
                    copied_path = input_img_tmpdir_path / png_path.name
                    shutil.copy(png_path, copied_path)
                    copied_png_paths_for_yomitoku.append(copied_path)
                    original_indices_map[copied_path.name] = original_page_index 
                except (ValueError, IndexError, Exception) as e:
                    st.warning(f"OCR用画像 {png_path.name} のコピーまたはインデックス抽出に失敗: {e}")
                    continue
            
            if not copied_png_paths_for_yomitoku:
                st.warning("OCR対象の画像がありません（コピー失敗など）。")
                return {}

            cmd = [
                "yomitoku", str(input_img_tmpdir_path),
                "-f", "md", "-o", str(ocr_output_tmpdir_path),
                "--device", device,
                "--combine", "--lite"
            ]
            try:
                process = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
            except subprocess.CalledProcessError as e:
                st.error(f"YomiTokuの実行に失敗しました。コマンド: {' '.join(cmd)}")
                st.error(f"エラー出力:\n{e.stderr}")
                return {}
            except FileNotFoundError:
                st.error("YomiTokuコマンドが見つかりません。インストールされているか、PATHが通っているか確認してください。")
                return {}

            md_files = list(ocr_output_tmpdir_path.glob("*.md"))
            if not md_files:
                st.warning("YomiTokuによるOCR結果のMarkdownファイルが見つかりませんでした。")
                return {}
                
            md_text = md_files[0].read_text(encoding="utf-8")
            parts = [s.strip() for s in md_text.split("\\n---\\n")]
            
            sorted_copied_png_names = sorted([p.name for p in copied_png_paths_for_yomitoku])

            if len(parts) != len(sorted_copied_png_names):
                st.warning(f"OCR結果のパーツ数({len(parts)})と画像数({len(sorted_copied_png_names)})が一致しません。")
            
            for i, text_part in enumerate(parts):
                if i < len(sorted_copied_png_names):
                    png_filename = sorted_copied_png_names[i]
                    if png_filename in original_indices_map:
                        original_idx = original_indices_map[png_filename]
                        md_by_page[original_idx] = text_part
                    else:
                        st.warning(f"OCR結果のファイル名 {png_filename} に対応する元のページインデックスが見つかりません。")
                else:
                    st.warning(f"OCR結果のパーツが画像数より多いです。パーツ {i+1} は無視されます。")
                    break 
    except Exception as e:
        st.error(f"OCR処理中（一時ディレクトリ管理など）に予期せぬエラー: {e}")
        return {}
    return md_by_page

# --------- 3. 単一 PDF 変換 (Concurrency and Cancellation Refactored) --------
def pdf_to_markdown(pdf_path: Path,
                    dst_dir: Path,
                    cache_dir: Path, 
                    device: str, 
                    filename_key: str, 
                    update_status_callback: Callable[[str, str], None]
                    ) -> None:
    # Point 1: At the very beginning
    if st.session_state.get('cancel_requested', False):
        update_status_callback(filename_key, "Cancelled")
        return
    
    update_status_callback(filename_key, "Processing...")
    doc = None 
    png_export_temp_dir = None # Initialize for finally block

    try:
        pdf_hash = sha256(pdf_path)
        cache_md = cache_dir / f"{pdf_hash}.md" 
        out_md  = dst_dir / f"{pdf_path.stem}.md"

        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            st.warning(f"Cache directory creation/check failed ({cache_dir}): {e}") 

        if cache_md.exists():
            try:
                shutil.copy(cache_md, out_md)
                update_status_callback(filename_key, f"Completed (cached) -> {out_md.name}")
                return
            except Exception as e:
                update_status_callback(filename_key, f"Error: Cache copy failed ({e}), retrying conversion.")

        try:
            doc = fitz.open(pdf_path)
        except Exception as e: 
            update_status_callback(filename_key, f"Error: Could not open PDF - {e}")
            return

        # Point 2: Before main OCR block
        if st.session_state.get('cancel_requested', False):
            update_status_callback(filename_key, "Cancelled")
            if doc: doc.close()
            return

        md_pages = [page_to_md(p) for p in doc]
        need_ocr_pages_indices = [i for i,p in enumerate(doc) if not has_text_layer(p)]

        if need_ocr_pages_indices:
            update_status_callback(filename_key, f"OCR for {len(need_ocr_pages_indices)} pages...")
            
            try:
                # Point 3: Inside OCR block, before export_pages_as_png
                if st.session_state.get('cancel_requested', False):
                    update_status_callback(filename_key, "Cancelled")
                    if doc: doc.close()
                    # No png_export_temp_dir to clean yet here
                    return

                png_export_temp_dir = Path(tempfile.mkdtemp(prefix=f"pdf2md_png_{filename_key}_"))
                update_status_callback(filename_key, f"OCR: Exporting {len(need_ocr_pages_indices)} pages to PNG...")
                
                # Check again after potentially long operation (temp dir creation)
                if st.session_state.get('cancel_requested', False):
                    update_status_callback(filename_key, "Cancelled")
                    if doc: doc.close()
                    if png_export_temp_dir and png_export_temp_dir.exists(): shutil.rmtree(png_export_temp_dir, ignore_errors=True)
                    return
                
                pngs  = export_pages_as_png(doc, need_ocr_pages_indices, dpi=220, outdir=png_export_temp_dir)

                # Point 4: Inside OCR block, before run_yomitoku
                if st.session_state.get('cancel_requested', False):
                    update_status_callback(filename_key, "Cancelled")
                    if doc: doc.close()
                    if png_export_temp_dir and png_export_temp_dir.exists(): shutil.rmtree(png_export_temp_dir, ignore_errors=True)
                    return

                if pngs:
                     update_status_callback(filename_key, f"OCR: Running YomiToku on {len(pngs)} image(s)...")
                     ocr_md_parts = run_yomitoku(pngs, device=device) # This is a blocking call
                     
                     # Check after blocking call
                     if st.session_state.get('cancel_requested', False):
                        update_status_callback(filename_key, "Cancelled")
                        # Resources cleaned in finally
                        return

                     update_status_callback(filename_key, f"OCR: Processing {len(ocr_md_parts)} text result(s) from YomiToku...")
                     for idx_in_doc, md_text_part in ocr_md_parts.items():
                         if 0 <= idx_in_doc < len(md_pages):
                             md_pages[idx_in_doc] = md_text_part
                         else:
                             st.warning(f"OCR result index {idx_in_doc} out of bounds for {filename_key}.")
                else:
                    st.warning(f"{filename_key}: No images exported for OCR, or export failed.")
            except Exception as e:
                update_status_callback(filename_key, f"Error during OCR processing: {e}")
            finally: # Ensures cleanup if OCR block is entered
                if png_export_temp_dir and png_export_temp_dir.exists():
                    try:
                        shutil.rmtree(png_export_temp_dir)
                    except Exception as e:
                        st.warning(f"Failed to remove temp PNG export dir ({png_export_temp_dir}): {e}")
        
        # Point 5: Before out_md.write_text
        if st.session_state.get('cancel_requested', False):
            update_status_callback(filename_key, "Cancelled")
            if doc: doc.close()
            return

        final_md = "\\n\\n---\\n\\n".join(md_pages)
        
        try:
            out_md.write_text(final_md, encoding="utf-8")
        except Exception as e:
            update_status_callback(filename_key, f"Error: Failed to write Markdown file - {e}")
            return

        try: 
            cache_md.write_text(final_md, encoding="utf-8")
        except Exception as e:
            st.warning(f"Failed to save Markdown to cache ({cache_md}): {e}")

        update_status_callback(filename_key, f"Completed -> {out_md.name}")

    except Exception as e: 
        update_status_callback(filename_key, f"Error: Unexpected failure - {e}")
    finally:
        if doc: 
            doc.close()
        # Cleanup OCR temp dir if exception occurred before its specific finally block
        if png_export_temp_dir and png_export_temp_dir.exists(): 
            try: shutil.rmtree(png_export_temp_dir, ignore_errors=True)
            except Exception : pass


# --------- 4. Streamlit GUI (Concurrency and Cancellation Refactored) ------------------------------------

def select_folder_dialog():
    root = tk.Tk()
    root.withdraw() 
    root.attributes('-topmost', True) 
    folder_selected = filedialog.askdirectory()
    root.destroy()
    return folder_selected

st.set_page_config(page_title="PDF to Markdown Converter", layout="wide")
st.title("📄 PDF to Markdown 一括変換ツール")

st.sidebar.header("設定")
uploaded_files = st.sidebar.file_uploader("PDFファイルを選択 (複数可)", type="pdf", accept_multiple_files=True)

st.sidebar.markdown("---")
st.sidebar.subheader("またはフォルダを指定")

if 'folder_path_for_text_input' not in st.session_state:
    st.session_state.folder_path_for_text_input = ""

if st.sidebar.button("フォルダを選択してパスを入力", key="select_folder_button"):
    selected_path = select_folder_dialog()
    if selected_path:
        st.session_state.folder_path_for_text_input = selected_path
    st.rerun()

folder_path_str = st.sidebar.text_input(
    "PDFが含まれるフォルダのパスを入力",
    value=st.session_state.folder_path_for_text_input, 
    help="上のボタンで選択するか、ここに直接パスを入力または貼り付けしてください。"
)

st.sidebar.markdown("---") 
st.sidebar.subheader("出力先フォルダ")

if 'dst_folder_path' not in st.session_state:
    st.session_state.dst_folder_path = str(Path.home() / "Documents" / "pdf2md_output") 

if st.sidebar.button("出力先フォルダを選択", key="select_dst_folder_button"):
    selected_dst_path = select_folder_dialog()
    if selected_dst_path:
        st.session_state.dst_folder_path = selected_dst_path
    st.rerun()

st.sidebar.caption(f"現在の出力先: {st.session_state.dst_folder_path}")

device_options = ["cpu"]
if shutil.which("nvidia-smi"): 
    device_options.insert(0, "cuda") 
device_default_index = 0 

device = st.sidebar.selectbox("OCRデバイス", device_options, index=device_default_index) 

cache_dir_global = Path(".mdcache_gui") 

# --- Session State Initialization for Concurrency & Cancellation ---
if 'file_statuses' not in st.session_state: st.session_state.file_statuses = {} 
if 'conversion_running' not in st.session_state: st.session_state.conversion_running = False
if 'current_processing_list' not in st.session_state: st.session_state.current_processing_list = []
if 'upload_temp_dir_session' not in st.session_state: st.session_state.upload_temp_dir_session = None
if 'tasks_submitted_this_job' not in st.session_state: st.session_state.tasks_submitted_this_job = False
if 'current_job_futures' not in st.session_state: st.session_state.current_job_futures = {}
if 'error_details_list' not in st.session_state: st.session_state.error_details_list = []
if 'cancel_requested' not in st.session_state: st.session_state.cancel_requested = False


# --- UI Placeholders ---
overall_progress_area = st.empty()
status_details_area = st.container()
cancel_button_area = st.empty() # Placeholder for cancel button

# --- "変換開始" (Start Conversion) Button Logic ---
if st.sidebar.button("変換開始", type="primary", key="start_conversion_concurrent_button"):
    st.session_state.conversion_running = True
    st.session_state.file_statuses = {} 
    st.session_state.current_processing_list = []
    st.session_state.tasks_submitted_this_job = False 
    st.session_state.current_job_futures = {} 
    st.session_state.error_details_list = [] 
    st.session_state.cancel_requested = False # Reset cancel flag for new job
    
    if st.session_state.upload_temp_dir_session and Path(st.session_state.upload_temp_dir_session).exists():
        try: shutil.rmtree(st.session_state.upload_temp_dir_session)
        except Exception as e: st.warning(f"Previous temp upload directory cleanup failed: {e}")
        st.session_state.upload_temp_dir_session = None

    local_pdf_paths_to_process = [] 
    current_folder_path_from_ui = folder_path_str 

    if current_folder_path_from_ui:
        if os.path.isdir(current_folder_path_from_ui):
            folder_path = Path(current_folder_path_from_ui)
            local_pdf_paths_to_process = sorted(list(folder_path.rglob("*.pdf")))
            if not local_pdf_paths_to_process:
                st.warning(f"No PDF files found in '{current_folder_path_from_ui}'.")
                st.session_state.conversion_running = False; st.stop()
            st.info(f"Processing PDFs from folder: {len(local_pdf_paths_to_process)} files.")
        else:
            st.error(f"Invalid folder path: '{current_folder_path_from_ui}'.")
            st.session_state.conversion_running = False; st.stop()
    elif uploaded_files:
        st.session_state.upload_temp_dir_session = tempfile.mkdtemp(prefix="pdf2md_gui_uploads_")
        upload_dir_path = Path(st.session_state.upload_temp_dir_session)
        for uploaded_file_data in uploaded_files:
            try:
                temp_pdf_path = upload_dir_path / uploaded_file_data.name
                with open(temp_pdf_path, "wb") as f: f.write(uploaded_file_data.getbuffer())
                local_pdf_paths_to_process.append(temp_pdf_path)
            except Exception as e: st.error(f"Failed to save uploaded file {uploaded_file_data.name}: {e}")
        if not local_pdf_paths_to_process:
            st.error("No PDF files were successfully saved from upload.")
            st.session_state.conversion_running = False; st.stop()
        st.info(f"Processing uploaded PDF files: {len(local_pdf_paths_to_process)} files.")
    else:
        st.sidebar.warning("Please select PDF files or specify a folder path.")
        st.session_state.conversion_running = False; st.stop()

    dst_dir_str_val = st.session_state.dst_folder_path 
    if not dst_dir_str_val:
        st.sidebar.error("Output folder is not selected.")
        st.session_state.conversion_running = False; st.stop()
    dst_dir_path_val = Path(dst_dir_str_val) 
    try: dst_dir_path_val.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        st.error(f"Failed to create output folder '{dst_dir_path_val}': {e}")
        st.session_state.conversion_running = False; st.stop()

    try: cache_dir_global.mkdir(parents=True, exist_ok=True)
    except Exception as e: st.warning(f"Failed to create cache folder '{cache_dir_global}': {e}") 
    
    st.info(f"Output will be saved to: {dst_dir_path_val}")
    st.info(f"Using OCR device: {device}")

    st.session_state.current_processing_list = local_pdf_paths_to_process
    for p_path_obj in local_pdf_paths_to_process: 
        st.session_state.file_statuses[p_path_obj.name] = "Pending" 
    
    st.rerun()

# --- Concurrent Processing & UI Update Logic ---
if st.session_state.conversion_running and st.session_state.current_processing_list:
    files_for_this_run_paths = st.session_state.current_processing_list 
    total_files_in_job = len(files_for_this_run_paths)

    # Display Cancel button
    if cancel_button_area.button("キャンセル (Cancel Conversion)", key="cancel_button_main_area"):
        st.session_state.cancel_requested = True
        st.warning("Cancellation requested. Will stop after current operations complete...")
        # Immediately mark pending files as Cancelled
        for fname_k, status_v in st.session_state.file_statuses.items():
            if status_v == "Pending":
                st.session_state.file_statuses[fname_k] = "Cancelled"
        st.rerun()

    def update_status_callback_main(filename_key: str, status_message: str):
        st.session_state.file_statuses[filename_key] = status_message

    current_dst_dir_path = Path(st.session_state.dst_folder_path) 

    if not st.session_state.tasks_submitted_this_job and \
       total_files_in_job > 0 and \
       not st.session_state.get('cancel_requested', False):
        cpu_cores_count = os.cpu_count()
        max_w = min(4, (cpu_cores_count if cpu_cores_count else 1) + 2)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_w) as executor:
            for pdf_path_for_task in files_for_this_run_paths: 
                if st.session_state.get('cancel_requested', False): 
                    if st.session_state.file_statuses.get(pdf_path_for_task.name) == "Pending":
                         st.session_state.file_statuses[pdf_path_for_task.name] = "Cancelled"
                    continue 
                future = executor.submit(pdf_to_markdown, 
                                         pdf_path_for_task, current_dst_dir_path, cache_dir_global, 
                                         device, pdf_path_for_task.name, update_status_callback_main)
                st.session_state.current_job_futures[future] = pdf_path_for_task.name
        st.session_state.tasks_submitted_this_job = True
    elif st.session_state.get('cancel_requested', False) and not st.session_state.tasks_submitted_this_job:
        # If cancel was hit before any tasks were submitted, mark all as cancelled.
        for p_path_obj_cancel in files_for_this_run_paths:
            if st.session_state.file_statuses.get(p_path_obj_cancel.name) == "Pending":
                st.session_state.file_statuses[p_path_obj_cancel.name] = "Cancelled"

    completed_count = 0
    error_count = 0
    cancelled_count = 0 
    current_job_filenames_list = [p.name for p in files_for_this_run_paths]

    for fname_key in current_job_filenames_list:
        status = st.session_state.file_statuses.get(fname_key, "Unknown")
        if "completed" in status.lower(): completed_count += 1
        elif "error" in status.lower(): error_count += 1
        elif status == "Cancelled": cancelled_count += 1
    
    if total_files_in_job > 0:
        progress_fraction = (completed_count + error_count + cancelled_count) / total_files_in_job
        overall_progress_area.progress(progress_fraction, 
            text=f"Progress: {completed_count+error_count+cancelled_count}/{total_files_in_job} files. C:{completed_count}, E:{error_count}, X:{cancelled_count} ({int(progress_fraction*100)}%)")

    with status_details_area.container():
        st.subheader("Individual File Statuses:")
        sorted_filenames_to_display = sorted(current_job_filenames_list)
        num_cols_to_display = 2 
        cols_ui = st.columns(num_cols_to_display)
        col_idx_ui = 0
        for filename_to_disp in sorted_filenames_to_display:
            status_to_disp = st.session_state.file_statuses.get(filename_to_disp, "Pending")
            with cols_ui[col_idx_ui % num_cols_to_display]:
                if "error" in status_to_disp.lower(): st.error(f"📄 {filename_to_disp}: {status_to_disp}")
                elif "completed" in status_to_disp.lower(): st.success(f"📄 {filename_to_disp}: {status_to_disp}")
                elif status_to_disp == "Cancelled": st.warning(f"📄 {filename_to_disp}: {status_to_disp}")
                elif "pending" in status_to_disp.lower(): st.caption(f"📄 {filename_to_disp}: {status_to_disp}")
                else: st.info(f"📄 {filename_to_disp}: {status_to_disp}")
            col_idx_ui += 1

    if (completed_count + error_count + cancelled_count) == total_files_in_job and total_files_in_job > 0:
        st.session_state.conversion_running = False 
        cancel_button_area.empty() # Clear cancel button
        st.session_state.error_details_list = [] 

        for fname, fstatus in st.session_state.file_statuses.items():
            if fname in current_job_filenames_list and "error" in fstatus.lower():
                if not any(d['file'] == fname for d in st.session_state.error_details_list):
                    st.session_state.error_details_list.append({"file": fname, "message": fstatus})

        futures_dict_check = st.session_state.get('current_job_futures', {})
        for future_obj_check, name_key_check in futures_dict_check.items():
            if name_key_check in current_job_filenames_list: 
                if future_obj_check.done() and future_obj_check.exception() is not None:
                    error_msg_uncaught = f"Error: Uncaught - {future_obj_check.exception()}"
                    if "error" not in st.session_state.file_statuses.get(name_key_check, "").lower():
                         st.session_state.file_statuses[name_key_check] = error_msg_uncaught
                    if not any(d['file'] == name_key_check for d in st.session_state.error_details_list):
                        st.session_state.error_details_list.append({"file": name_key_check, "message": error_msg_uncaught})
                    else: 
                        for item in st.session_state.error_details_list:
                            if item['file'] == name_key_check and "Uncaught Exception" not in item['message']:
                                item['message'] = error_msg_uncaught; break
        
        final_error_tally = len(st.session_state.error_details_list)
        # Recalculate completed based on final error tally and known cancelled
        final_completed_tally = total_files_in_job - final_error_tally - cancelled_count 

        if st.session_state.error_details_list:
            with st.expander(f"Error Summary ({final_error_tally} file(s) failed)", expanded=True):
                for err_info in st.session_state.error_details_list:
                    st.error(f"File: {err_info['file']} - Details: {err_info['message']}")
        
        if st.session_state.get('cancel_requested', False):
            st.warning(f"Conversion process cancelled. Processed: {final_completed_tally} completed, {final_error_tally} errors, {cancelled_count} explicitly cancelled.")
        elif final_error_tally == 0 and final_completed_tally == total_files_in_job: # No errors, no cancellations subtracted
            st.balloons()
            st.success(f"All {total_files_in_job} files converted successfully!")
        else: # Handles cases with errors, or completed with some errors and no explicit cancel
             st.info(f"Conversion finished. Results: {final_completed_tally} completed, {final_error_tally} errors, {cancelled_count} cancelled.")
        
        if st.session_state.upload_temp_dir_session and Path(st.session_state.upload_temp_dir_session).exists():
            try: shutil.rmtree(st.session_state.upload_temp_dir_session)
            except Exception as e: st.warning(f"Failed to clean up temporary upload directory: {e}")
            st.session_state.upload_temp_dir_session = None
        
        st.session_state.current_job_futures = {} 

    elif st.session_state.conversion_running : 
        time.sleep(1.0) 
        st.rerun() 

if not st.session_state.conversion_running and st.session_state.file_statuses:
    if st.button("Clear Previous Results"):
        st.session_state.file_statuses = {}
        st.session_state.current_processing_list = []
        st.session_state.current_job_futures = {}
        st.session_state.tasks_submitted_this_job = False
        st.session_state.error_details_list = []
        st.session_state.cancel_requested = False
        overall_progress_area.empty() 
        status_details_area.empty() 
        cancel_button_area.empty()
        st.rerun()

st.markdown("---")
st.markdown("""
### 使い方のヒント
1.  左のサイドバーから、個別のPDFファイルを選択するか、PDFが含まれるフォルダのパスを指定します。
    （フォルダパスを指定した場合、アップロードされたファイルは無視されます。）
2.  Markdownファイルの出力先フォルダを指定します（存在しない場合は作成されます）。
3.  OCRに使用するデバイスを選択します（CUDA対応GPUがあれば `cuda` を、なければ `cpu` を選択）。
4.  「変換開始」ボタンを押すと、処理が始まります。処理中は進捗が表示されます。
5.  処理中に「キャンセル」ボタンを押すと、新規タスクの開始を停止し、現在進行中のタスク完了後に処理を終えます。
""")
