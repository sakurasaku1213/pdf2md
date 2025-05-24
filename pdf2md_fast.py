# pdf2md_fast.py (Streamlit GUI版)
# ------------------------------------------------------------
# 依存:
#   pip install "pymupdf<1.25" streamlit yomi-toku
#   # OCR フォールバック用 – CUDA or CPU で動作
#   # Poppler が不要な純 PyMuPDF ルート
# ------------------------------------------------------------
import os, sys, json, hashlib, shutil, subprocess, tempfile
# concurrent.futures はStreamlitのシンプルなGUIでは直接使わず、ファイルごとに処理します
from pathlib import Path
from typing import List, Dict

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
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            if not line["spans"]: # Add check for empty line spans
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

# --------- 2. OCR フォールバック (一時ディレクトリ処理を改善) -------------
def export_pages_as_png(doc: fitz.Document,
                        indices: List[int],
                        dpi: int = 220,
                        outdir: Path = None) -> List[Path]:
    if outdir is None:
        # Ensure the temporary directory for PNGs is created successfully
        try:
            outdir = Path(tempfile.mkdtemp(prefix="pdf2md_png_"))
        except Exception as e:
            st.error(f"PNGエクスポート用の一時ディレクトリ作成に失敗: {e}")
            return [] # Return empty list if directory creation fails
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
    # Use try-except for TemporaryDirectory context managers
    try:
        with tempfile.TemporaryDirectory(prefix="yomitoku_ocr_out_") as ocr_output_tmpdir_str, \
             tempfile.TemporaryDirectory(prefix="yomitoku_input_img_") as input_img_tmpdir_str:
            
            ocr_output_tmpdir_path = Path(ocr_output_tmpdir_str)
            input_img_tmpdir_path = Path(input_img_tmpdir_str)
                
            copied_png_paths_for_yomitoku = []
            original_indices_map = {} # Maps copied filename back to original page index

            for png_path in png_paths:
                try:
                    # Extract original page index from filename like "page_XX.png"
                    original_page_index = int(png_path.stem.split('_')[1]) - 1
                    copied_path = input_img_tmpdir_path / png_path.name
                    shutil.copy(png_path, copied_path)
                    copied_png_paths_for_yomitoku.append(copied_path)
                    # Store mapping from the copied image's name to its original document page index
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
                # Ensure stderr is captured and decoded properly
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
            # YomiToku combine 出力はページ区切りに '---' を使う
            parts = [s.strip() for s in md_text.split("\\n---\\n")] # Adjusted split pattern
            
            # Sort the copied PNG names to match the order of 'parts' from YomiToku's combined output
            sorted_copied_png_names = sorted([p.name for p in copied_png_paths_for_yomitoku])

            if len(parts) != len(sorted_copied_png_names):
                st.warning(f"OCR結果のパーツ数({len(parts)})と画像数({len(sorted_copied_png_names)})が一致しません。処理結果が不正確になる可能性があります。")
                # Attempt to process what we can, or return {}
            
            for i, text_part in enumerate(parts):
                if i < len(sorted_copied_png_names):
                    png_filename = sorted_copied_png_names[i]
                    if png_filename in original_indices_map:
                        original_idx = original_indices_map[png_filename]
                        md_by_page[original_idx] = text_part
                    else:
                        st.warning(f"OCR結果のファイル名 {png_filename} に対応する元のページインデックスが見つかりません。")
                else:
                    # More parts than images, something is wrong
                    st.warning(f"OCR結果のパーツが画像数より多いです。パーツ {i+1} は無視されます。")
                    break 
    except Exception as e:
        st.error(f"OCR処理中（一時ディレクトリ管理など）に予期せぬエラー: {e}")
        return {} # Return empty if any critical error in temp dir handling
    return md_by_page

# --------- 3. 単一 PDF 変換 (エラー処理と一時ディレクトリ管理を強化) --------
def pdf_to_markdown(pdf_path: Path,
                    dst_dir: Path,
                    cache_dir: Path,
                    device: str = "cuda",
                    progress_bar=None,
                    file_idx=0,
                    total_files=1
                    ) -> None:
    if progress_bar:
        progress_text = f"処理中: {pdf_path.name} ({file_idx+1}/{total_files})"
        try:
            progress_value = (file_idx / total_files) if total_files > 0 else 0
            progress_bar.progress(progress_value, text=progress_text)
        except Exception as e:
            st.warning(f"プログレスバーの更新に失敗: {e}")


    pdf_hash = sha256(pdf_path)
    cache_md = cache_dir / f"{pdf_hash}.md"
    out_md  = dst_dir / f"{pdf_path.stem}.md"

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        st.warning(f"キャッシュディレクトリの作成/確認に失敗 ({cache_dir}): {e}")
        # Continue without cache if it fails

    if cache_md.exists():
        try:
            shutil.copy(cache_md, out_md)
            st.info(f"キャッシュを利用しました: {pdf_path.name} -> {out_md.name}")
            if progress_bar:
                 new_progress = ((file_idx + 1) / total_files) if total_files > 0 else 1
                 progress_bar.progress(new_progress, text=f"完了 (キャッシュ): {pdf_path.name} ({file_idx+1}/{total_files})")
            return
        except Exception as e:
            st.warning(f"キャッシュファイルのコピーに失敗 ({cache_md} -> {out_md}): {e}。通常変換を試みます。")

    doc = None
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        st.error(f"PDFファイルを開けませんでした: {pdf_path.name} - {e}")
        if progress_bar:
            new_progress = ((file_idx + 1) / total_files) if total_files > 0 else 1
            progress_bar.progress(new_progress, text=f"エラー: {pdf_path.name} ({file_idx+1}/{total_files})")
        return

    md_pages = [page_to_md(p) for p in doc] # Initial conversion from text layer
    need_ocr_pages_indices = [i for i,p in enumerate(doc) if not has_text_layer(p)]

    if need_ocr_pages_indices:
        st.write(f"{pdf_path.name}: {len(need_ocr_pages_indices)} ページでOCRを実行します...")
        png_export_temp_dir = None # Initialize
        try:
            png_export_temp_dir = Path(tempfile.mkdtemp(prefix="pdf2md_gui_png_export_"))
            pngs  = export_pages_as_png(doc, need_ocr_pages_indices, dpi=220, outdir=png_export_temp_dir)
            if pngs: # If any PNGs were successfully exported
                 ocr_md_parts = run_yomitoku(pngs, device=device)
                 for idx_in_doc, md_text_part in ocr_md_parts.items():
                     if 0 <= idx_in_doc < len(md_pages): # Check index bounds
                         md_pages[idx_in_doc] = md_text_part # Overwrite with OCR text
                     else:
                         st.warning(f"OCR結果のインデックス {idx_in_doc} がページ範囲外です ({pdf_path.name})。")
            else:
                st.warning(f"{pdf_path.name}: OCR対象ページの画像エクスポートに失敗、または対象画像がありませんでした。")
        except Exception as e:
            st.error(f"OCR処理中にエラーが発生 ({pdf_path.name}): {e}")
        finally:
            if png_export_temp_dir and png_export_temp_dir.exists():
                try:
                    shutil.rmtree(png_export_temp_dir)
                except Exception as e:
                    st.warning(f"PNGエクスポート用一時ディレクトリの削除に失敗 ({png_export_temp_dir}): {e}")
            
    final_md = "\\n\\n---\\n\\n".join(md_pages) # Page separator for final markdown
    try:
        out_md.write_text(final_md, encoding="utf-8")
        try:
            # Attempt to save to cache even if main write succeeds
            cache_md.write_text(final_md, encoding="utf-8")
        except Exception as e:
            st.warning(f"Markdownのキャッシュ保存に失敗 ({cache_md}): {e}")
        st.success(f"変換完了: {pdf_path.name} -> {out_md.name}")
    except Exception as e:
        st.error(f"Markdownファイルの書き出しに失敗 ({out_md}): {e}")
        if progress_bar:
            new_progress = ((file_idx + 1) / total_files) if total_files > 0 else 1
            progress_bar.progress(new_progress, text=f"エラー(書き出し失敗): {pdf_path.name} ({file_idx+1}/{total_files})")
        # Ensure doc is closed even if write fails, if it was opened
        if doc:
            doc.close()
        return

    if progress_bar:
        new_progress = ((file_idx + 1) / total_files) if total_files > 0 else 1
        progress_bar.progress(new_progress, text=f"完了: {pdf_path.name} ({file_idx+1}/{total_files})")
    
    if doc: # Close the document
        doc.close()

# --------- 4. Streamlit GUI -------------------------------------------------

def select_folder_dialog():
    """フォルダ選択ダイアログを開き、選択されたフォルダパスを返す"""
    root = tk.Tk()
    root.withdraw()  # Tkinterのメインウィンドウを表示しない
    root.attributes('-topmost', True)  # ダイアログを最前面に表示
    folder_selected = filedialog.askdirectory()
    root.destroy()
    return folder_selected

st.set_page_config(page_title="PDF to Markdown Converter", layout="wide")
st.title("📄 PDF to Markdown 一括変換ツール")

st.sidebar.header("設定")
uploaded_files = st.sidebar.file_uploader("PDFファイルを選択 (複数可)", type="pdf", accept_multiple_files=True)

st.sidebar.markdown("---")
st.sidebar.subheader("またはフォルダを指定")

# セッションステートでフォルダパスを管理
if 'folder_path_for_text_input' not in st.session_state:
    st.session_state.folder_path_for_text_input = ""

if st.sidebar.button("フォルダを選択してパスを入力", key="select_folder_button"):
    selected_path = select_folder_dialog()
    if selected_path:
        st.session_state.folder_path_for_text_input = selected_path
    # ボタンが押されたら一度スクリプトを再実行してテキスト入力に反映させる
    st.rerun()


folder_path_str = st.sidebar.text_input(
    "PDFが含まれるフォルダのパスを入力",
    value=st.session_state.folder_path_for_text_input, # セッションステートから値を取得
    help="上のボタンで選択するか、ここに直接パスを入力または貼り付けしてください。例: D:\\\\scanned_documents (サブフォルダも検索します)"
)

# --- 出力先フォルダ選択 ---
st.sidebar.markdown("---") # 区切り線
st.sidebar.subheader("出力先フォルダ")

if 'dst_folder_path' not in st.session_state:
    st.session_state.dst_folder_path = str(Path.home() / "Documents" / "pdf2md_output") # 初期値を設定

if st.sidebar.button("出力先フォルダを選択", key="select_dst_folder_button"):
    selected_dst_path = select_folder_dialog()
    if selected_dst_path:
        st.session_state.dst_folder_path = selected_dst_path
    st.rerun()

# 選択された出力先フォルダパスを表示 (編集不可)
st.sidebar.caption(f"現在の出力先: {st.session_state.dst_folder_path}")


device_options = ["cpu"]
if shutil.which("nvidia-smi"): # Check if nvidia-smi (CUDA utility) is available
    device_options.insert(0, "cuda") # Add cuda as first option if available
device_default_index = 0 # Default to first option (cuda if available, else cpu)

device = st.sidebar.selectbox("OCRデバイス", device_options, index=device_default_index) 

cache_dir = Path(".mdcache_gui") # GUI-specific cache directory

if st.sidebar.button("変換開始", type="primary", key="start_conversion_button"):
    pdf_paths_to_process = []
    source_type = None

    # folder_path_str に st.session_state の最新値を代入し直す (ボタン経由の場合を考慮)
    current_folder_path_from_input = folder_path_str

    # 1. Determine the source of PDF files
    if current_folder_path_from_input: # テキスト入力フィールドの値を使用
        # folder_path = Path(folder_path_str) # Keep for rglob, but check with os.path.isdir
        if os.path.isdir(current_folder_path_from_input): # Use os.path.isdir for initial validation
            folder_path = Path(current_folder_path_from_input) # Convert to Path after validation for rglob
            # Use rglob for recursive search and sort the results
            pdf_paths_to_process = sorted(list(folder_path.rglob("*.pdf")))
            if not pdf_paths_to_process:
                st.warning(f"指定フォルダ '{current_folder_path_from_input}' (サブフォルダ含む) にPDFファイルが見つかりません。")
                st.stop() # Use st.stop() to halt execution cleanly
            source_type = "folder"
            st.info(f"フォルダ '{current_folder_path_from_input}' 内のPDFを処理します ({len(pdf_paths_to_process)}件)。")
        else:
            st.error(f"指定されたパス '{current_folder_path_from_input}' は有効なフォルダではありません。")
            st.stop()
    elif uploaded_files:
        source_type = "upload"
        st.info(f"アップロードされた {len(uploaded_files)}個のPDFファイルを処理します。")
    else:
        st.sidebar.warning("PDFファイルを選択するか、フォルダパスを指定してください。")
        st.stop()

    # 2. Validate destination directory
    dst_dir_str = st.session_state.dst_folder_path # セッションステートから取得
    if not dst_dir_str:
        st.sidebar.error("出力先フォルダが選択されていません。") # エラーメッセージに変更
        st.stop()
    dst_dir = Path(dst_dir_str)
    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        st.error(f"出力先フォルダの作成に失敗: {dst_dir} - {e}")
        st.stop()

    # 3. Setup cache
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        st.warning(f"キャッシュフォルダの作成に失敗 ({cache_dir}): {e}")
    
    st.info(f"出力先フォルダ: {dst_dir}")
    st.info(f"OCRデバイス: {device}")

    # 4. Process PDFs
    progress_bar_area = st.empty() # Placeholder for the progress bar

    if source_type == "upload":
        # Use a context manager for the temporary directory for uploads
        with tempfile.TemporaryDirectory(prefix="pdf2md_gui_upload_") as upload_tmpdir_str:
            upload_tmpdir_path = Path(upload_tmpdir_str)
            temp_pdf_paths_from_upload = [] # Store paths of successfully saved temp files
            for uploaded_file_data in uploaded_files:
                try:
                    temp_pdf_path = upload_tmpdir_path / uploaded_file_data.name
                    with open(temp_pdf_path, "wb") as f:
                        f.write(uploaded_file_data.getbuffer())
                    temp_pdf_paths_from_upload.append(temp_pdf_path)
                except Exception as e:
                    st.error(f"アップロードファイル {uploaded_file_data.name} の一時保存失敗: {e}")
            
            if not temp_pdf_paths_from_upload: # If no files were successfully saved
                st.error("処理対象のPDFファイルがありません（一時保存失敗）。")
                progress_bar_area.empty() # Clear progress bar area
                st.stop()
            
            pdf_paths_to_process = temp_pdf_paths_from_upload # Update the list to process

    # This block will now execute for both 'folder' and 'upload' (after uploads are prepared)
    if pdf_paths_to_process: # Ensure there are files to process
        total_files = len(pdf_paths_to_process)
        progress_bar = progress_bar_area.progress(0, text=f"準備中... (0/{total_files})")
        
        for i, pdf_path_item in enumerate(pdf_paths_to_process):
            pdf_to_markdown(pdf_path_item, dst_dir, cache_dir, device, progress_bar, i, total_files)
        
        progress_bar_area.empty() # Clear progress bar after completion
        st.balloons()
        st.success(f"すべてのファイルの変換が完了しました！ ({total_files}件処理)")
    else:
        # This case should ideally be caught earlier, but as a fallback:
        st.warning("処理対象のPDFファイルが見つかりませんでした。")


st.markdown("---")
st.markdown("""
### 使い方のヒント
1.  左のサイドバーから、個別のPDFファイルを選択するか、PDFが含まれるフォルダのパスを指定します。
    （フォルダパスを指定した場合、アップロードされたファイルは無視されます。）
2.  Markdownファイルの出力先フォルダを指定します（存在しない場合は作成されます）。
3.  OCRに使用するデバイスを選択します（CUDA対応GPUがあれば `cuda` を、なければ `cpu` を選択）。
4.  「変換開始」ボタンを押すと、処理が始まります。
""")

# To run this script: streamlit run your_script_name.py
# Ensure Typer related app.run() or similar is removed if this was converted from a Typer CLI.
# The main execution flow is now handled by Streamlit's rendering of the script from top to bottom.
