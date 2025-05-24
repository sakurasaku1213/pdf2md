# pdf2md_fast_clean.py (Streamlit GUI版) - 超高速化CPU特化
# ==============================================================================
# 🚀 依存パッケージ:
#   pip install "pymupdf<1.25" streamlit yomi-toku concurrent-futures threadpoolctl
#   # CPU特化で最大パフォーマンス | 並列処理 | 高速OCR
# ==============================================================================
import os, sys, json, hashlib, shutil, subprocess, tempfile, time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import threading

import fitz                # PyMuPDF
import streamlit as st
import tkinter as tk
from tkinter import filedialog

# CPU最適化設定
os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count())
os.environ["OPENBLAS_NUM_THREADS"] = str(mp.cpu_count())

# ページ設定
st.set_page_config(
    page_title="PDF→Markdown 超高速変換",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================= 高速化ユーティリティ =========================================

@st.cache_data
def sha256_cached(file_path_str: str) -> str:
    """キャッシュ化されたハッシュ計算"""
    fp = Path(file_path_str)
    h = hashlib.sha256()
    with fp.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256(fp: Path) -> str:
    """高速ハッシュ計算（キャッシュ対応）"""
    return sha256_cached(str(fp))

# ========================================= 高速テキスト抽出 =========================================

_MIN_CHARS = 30

def has_text_layer(page: fitz.Page, min_chars: int = _MIN_CHARS) -> bool:
    """テキスト層の存在チェック（高速化）"""
    try:
        text = page.get_text()
        return len(text.strip()) >= min_chars
    except:
        return False

def page_to_md_fast(page: fitz.Page) -> str:
    """高速Markdown変換（最適化済み）"""
    try:
        # 高速テキスト抽出
        text_dict = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)
        if not text_dict or "blocks" not in text_dict:
            return ""
        
        lines = []
        for block in text_dict["blocks"]:
            if block.get("type") != 0:  # テキストブロックのみ
                continue
            
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                
                span = spans[0]
                text = span.get("text", "").rstrip()
                if text:
                    lines.append({
                        "size": span.get("size", 12),
                        "text": text
                    })
        
        if not lines:
            return ""
        
        # サイズ別見出し判定（高速化）
        sizes = sorted(set(l["size"] for l in lines), reverse=True)
        h1_size = sizes[0] if sizes else 12
        h2_size = sizes[1] if len(sizes) > 1 else h1_size
        
        # Markdown変換（最適化）
        md_lines = []
        bullet_prefixes = ("•", "・", "〇", "◯", "-", "―", "–", "*")
        
        for line in lines:
            txt = line["text"]
            if not txt:
                continue
                
            size = line["size"]
            if size >= h1_size:
                md_lines.append(f"# {txt}")
            elif size >= h2_size:
                md_lines.append(f"## {txt}")
            elif txt.lstrip().startswith(bullet_prefixes):
                clean_text = txt.lstrip('•・〇◯-–―* ')
                md_lines.append(f"- {clean_text}")
            else:
                md_lines.append(txt)
        
        return "\n".join(md_lines)
        
    except Exception as e:
        st.warning(f"ページのMarkdown変換でエラー: {e}")
        return ""

# 下位互換性のためのエイリアス
page_to_md = page_to_md_fast

# ========================================= 超高速OCR処理 =========================================

def export_pages_as_png_parallel(doc: fitz.Document, indices: List[int], 
                                dpi: int = 150, outdir: Path = None, 
                                max_workers: int = None) -> List[Path]:
    """並列画像エクスポート（DPI下げて高速化）"""
    if outdir is None:
        try:
            outdir = Path(tempfile.mkdtemp(prefix="pdf2md_png_fast_"))
        except Exception as e:
            st.error(f"PNGエクスポート用ディレクトリ作成失敗: {e}")
            return []
    else:
        outdir = Path(outdir)
    
    try:
        outdir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        st.error(f"PNGエクスポート先作成失敗: {e}")
        return []
    
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(indices))
    
    def export_single_page(page_idx: int) -> Optional[Path]:
        try:
            page = doc.load_page(page_idx)
            # 高速化: より小さなDPIと圧縮画像
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            png_path = outdir / f"page_{page_idx+1}.png"
            pix.save(png_path)
            pix = None  # メモリ解放
            return png_path
        except Exception as e:
            st.warning(f"ページ {page_idx+1} のPNGエクスポート失敗: {e}")
            return None
    
    png_paths = []
    # スレッドプールで並列処理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(export_single_page, idx): idx for idx in indices}
        for future in as_completed(future_to_idx):
            result = future.result()
            if result:
                png_paths.append(result)
    
    return sorted(png_paths)  # ページ順序を保持

def run_yomitoku_fast(png_paths: List[Path], device: str = "cpu") -> Dict[int, str]:
    """YomiToku高速実行（CPU最適化）"""
    if not png_paths:
        return {}
    
    md_by_page = {}
    
    try:
        with tempfile.TemporaryDirectory(prefix="yomitoku_fast_out_") as ocr_output_tmpdir_str, \
             tempfile.TemporaryDirectory(prefix="yomitoku_fast_in_") as input_img_tmpdir_str:
            
            ocr_output_tmpdir_path = Path(ocr_output_tmpdir_str)
            input_img_tmpdir_path = Path(input_img_tmpdir_str)
            
            # 高速ファイルコピー（並列）
            original_indices_map = {}
            
            def copy_png(png_path: Path) -> Optional[Tuple[Path, int]]:
                try:
                    original_page_index = int(png_path.stem.split('_')[1]) - 1
                    copied_path = input_img_tmpdir_path / png_path.name
                    shutil.copy2(png_path, copied_path)  # copy2は高速
                    return copied_path, original_page_index
                except Exception:
                    return None
            
            copied_paths = []
            with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
                results = list(executor.map(copy_png, png_paths))
                for result in results:
                    if result:
                        copied_path, original_idx = result
                        copied_paths.append(copied_path)
                        original_indices_map[copied_path.name] = original_idx
            
            if not copied_paths:
                return {}
            
            # YomiToku実行（CPU最適化設定）
            cmd = [
                "yomitoku", str(input_img_tmpdir_path),
                "-f", "md", "-o", str(ocr_output_tmpdir_path),
                "--device", device,
                "--combine", "--lite",
                "--batch-size", "4" if device == "cpu" else "8"  # CPU時はバッチサイズ小さく
            ]
            
            try:
                process = subprocess.run(
                    cmd, 
                    check=True, 
                    capture_output=True, 
                    text=True, 
                    encoding='utf-8', 
                    errors='replace',
                    timeout=300  # 5分タイムアウト
                )
            except subprocess.TimeoutExpired:
                st.error("OCR処理がタイムアウトしました（5分）")
                return {}
            except subprocess.CalledProcessError as e:
                st.error(f"YomiToku実行失敗: {e.stderr}")
                return {}
            except FileNotFoundError:
                st.error("YomiTokuコマンドが見つかりません")
                return {}
            
            # 結果読み込み
            md_files = list(ocr_output_tmpdir_path.glob("*.md"))
            if not md_files:
                return {}
                
            md_text = md_files[0].read_text(encoding="utf-8")
            parts = [s.strip() for s in md_text.split("\n---\n")]
            
            # 結果マッピング
            sorted_names = sorted([p.name for p in copied_paths])
            for i, text_part in enumerate(parts):
                if i < len(sorted_names):
                    png_filename = sorted_names[i]
                    if png_filename in original_indices_map:
                        original_idx = original_indices_map[png_filename]
                        md_by_page[original_idx] = text_part
                        
    except Exception as e:
        st.error(f"OCR処理中エラー: {e}")
        return {}
    
    return md_by_page

# ========================================= 超高速並列PDF変換 =========================================

def pdf_to_markdown_ultra_fast(pdf_path: Path, dst_dir: Path, cache_dir: Path, 
                               device: str = "cpu") -> Tuple[str, str]:
    """超高速PDF変換（並列処理）"""
    start_time = time.time()
    
    try:
        # キャッシュチェック（高速）
        pdf_hash = sha256(pdf_path)
        cache_md = cache_dir / f"{pdf_hash}.md"
        out_md = dst_dir / f"{pdf_path.stem}.md"
        
        if cache_md.exists():
            try:
                shutil.copy2(cache_md, out_md)
                elapsed = time.time() - start_time
                return "cached", f"⚡ キャッシュ利用 ({elapsed:.2f}秒)"
            except Exception:
                pass
        
        # PDF読み込み
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        
        # 並列テキスト抽出
        def extract_page_text(page_idx: int) -> Tuple[int, str, bool]:
            page = doc.load_page(page_idx)
            has_text = has_text_layer(page)
            if has_text:
                md_text = page_to_md_fast(page)
                return page_idx, md_text, True
            else:
                return page_idx, "", False
        
        # CPU並列処理
        max_workers = min(mp.cpu_count(), total_pages, 8)  # 最大8並列
        md_pages = [""] * total_pages
        need_ocr_indices = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(extract_page_text, i): i for i in range(total_pages)}
            for future in as_completed(future_to_idx):
                page_idx, md_text, has_text = future.result()
                if has_text:
                    md_pages[page_idx] = md_text
                else:
                    need_ocr_indices.append(page_idx)
        
        # OCR処理（必要な場合のみ）
        if need_ocr_indices:
            st.info(f"🔬 OCR実行: {len(need_ocr_indices)}ページ")
            
            with tempfile.TemporaryDirectory(prefix="pdf2md_ultra_fast_") as png_temp_dir:
                png_temp_path = Path(png_temp_dir)
                
                # 並列画像エクスポート
                pngs = export_pages_as_png_parallel(
                    doc, need_ocr_indices, 
                    dpi=150,  # 高速化のため低DPI
                    outdir=png_temp_path,
                    max_workers=max_workers
                )
                
                if pngs:
                    # OCR実行
                    ocr_results = run_yomitoku_fast(pngs, device=device)
                    for page_idx, ocr_text in ocr_results.items():
                        if 0 <= page_idx < total_pages:
                            md_pages[page_idx] = ocr_text
        
        # 最終結果統合
        final_md = "\n\n---\n\n".join(filter(None, md_pages))
        
        # 並列ファイル書き出し
        def write_file(path: Path, content: str) -> bool:
            try:
                path.write_text(content, encoding="utf-8")
                return True
            except:
                return False
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            output_future = executor.submit(write_file, out_md, final_md)
            cache_future = executor.submit(write_file, cache_md, final_md)
            
            output_success = output_future.result()
            cache_future.result()  # キャッシュは失敗しても問題なし
        
        doc.close()
        elapsed = time.time() - start_time
        
        if output_success:
            return "success", f"✅ 変換完了 ({elapsed:.2f}秒)"
        else:
            return "failed", f"❌ 書き出し失敗 ({elapsed:.2f}秒)"
            
    except Exception as e:
        return "failed", f"❌ 変換失敗: {e}"

def process_pdfs_ultra_fast(pdf_paths: List[Path], dst_dir: Path, cache_dir: Path, 
                           device: str = "cpu", progress_callback=None) -> Dict[str, int]:
    """並列PDF一括変換"""
    total_files = len(pdf_paths)
    results = {"success": 0, "cached": 0, "failed": 0}
    
    # キャッシュディレクトリ作成
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except:
        pass
    
    # 並列処理（CPUコア数に基づく）
    max_workers = min(mp.cpu_count() // 2, 4, total_files)  # メモリ使用量を考慮
    
    def process_single_pdf(args):
        idx, pdf_path = args
        result, message = pdf_to_markdown_ultra_fast(pdf_path, dst_dir, cache_dir, device)
        return idx, pdf_path.name, result, message
    
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # タスク提出
        future_to_args = {
            executor.submit(process_single_pdf, (i, pdf_path)): (i, pdf_path) 
            for i, pdf_path in enumerate(pdf_paths)
        }
        
        # 結果収集
        for future in as_completed(future_to_args):
            idx, filename, result, message = future.result()
            results[result] += 1
            completed += 1
            
            # プログレス更新
            if progress_callback:
                progress_value = completed / total_files
                progress_callback(progress_value, f"{message} | {filename} ({completed}/{total_files})")
            
            # ログ出力
            if result == "success":
                st.success(message + f" | {filename}")
            elif result == "cached":
                st.info(message + f" | {filename}")
            else:
                st.error(message + f" | {filename}")
    
    return results

# ========================================= Streamlit GUI =========================================

def select_folder_dialog():
    """フォルダ選択ダイアログを開き、選択されたフォルダパスを返す"""
    root = tk.Tk()
    root.withdraw()  # Tkinterのメインウィンドウを表示しない
    root.attributes('-topmost', True)  # ダイアログを最前面に表示
    folder_selected = filedialog.askdirectory()
    root.destroy()
    return folder_selected

# ========================================= メインGUI =========================================

# グラデーションヘッダー
st.markdown("""
<div style="background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1); 
            padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <h1 style="color: white; text-align: center; margin: 0;">
        ⚡ PDF→Markdown 超高速変換ツール
    </h1>
    <p style="color: white; text-align: center; margin: 10px 0 0 0;">
        CPU並列処理最適化 | 高速OCR | キャッシュ機能
    </p>
</div>
""", unsafe_allow_html=True)

# システム情報表示
cpu_count = mp.cpu_count()
st.sidebar.success(f"🖥️ CPU コア数: {cpu_count}")

# サイドバー設定
st.sidebar.header("⚙️ 変換設定")

# 入力モード選択
input_mode = st.sidebar.radio(
    "入力方式を選択",
    ["📁 フォルダ指定", "📄 ファイルアップロード"],
    help="フォルダから一括処理するか、個別ファイルをアップロードするかを選択"
)

if input_mode == "📄 ファイルアップロード":
    uploaded_files = st.sidebar.file_uploader(
        "PDFファイルを選択 (複数可)", 
        type="pdf", 
        accept_multiple_files=True,
        help="複数のPDFファイルを同時に選択できます"
    )
else:
    uploaded_files = None

if input_mode == "📁 フォルダ指定":
    st.sidebar.subheader("📂 フォルダ選択")
    
    # セッションステートでフォルダパスを管理
    if 'folder_path_for_text_input' not in st.session_state:
        st.session_state.folder_path_for_text_input = ""

    if st.sidebar.button("📁 フォルダを選択", key="select_folder_button"):
        selected_path = select_folder_dialog()
        if selected_path:
            st.session_state.folder_path_for_text_input = selected_path
        st.rerun()

    folder_path_str = st.sidebar.text_input(
        "フォルダパス",
        value=st.session_state.folder_path_for_text_input,
        help="PDFが含まれるフォルダのパス（サブフォルダも検索対象）"
    )
else:
    folder_path_str = ""

# 出力先フォルダ選択
st.sidebar.subheader("📤 出力設定")

if 'dst_folder_path' not in st.session_state:
    st.session_state.dst_folder_path = str(Path.home() / "Documents" / "pdf2md_output")

if st.sidebar.button("📤 出力先を選択", key="select_dst_folder_button"):
    selected_dst_path = select_folder_dialog()
    if selected_dst_path:
        st.session_state.dst_folder_path = selected_dst_path
    st.rerun()

st.sidebar.caption(f"📍 出力先: {st.session_state.dst_folder_path}")

# デバイス選択
device_options = ["cpu"]
if shutil.which("nvidia-smi"):
    device_options.insert(0, "cuda")

device = st.sidebar.selectbox(
    "🚀 OCRデバイス", 
    device_options, 
    index=0,
    help="CUDA対応GPUがある場合はcudaを選択（高速化）"
)

# パフォーマンス設定
st.sidebar.subheader("⚡ パフォーマンス")
st.sidebar.info(f"並列処理: 最大 {cpu_count} コア使用")

cache_dir = Path(".mdcache_gui")

# 変換開始ボタン
if st.sidebar.button("🚀 変換開始", type="primary", key="start_conversion_button"):
    pdf_paths_to_process = []
    source_type = None

    # 入力ファイル確認
    if input_mode == "📁 フォルダ指定" and folder_path_str:
        if os.path.isdir(folder_path_str):
            folder_path = Path(folder_path_str)
            pdf_paths_to_process = sorted(list(folder_path.rglob("*.pdf")))
            if not pdf_paths_to_process:
                st.warning(f"📂 フォルダ '{folder_path_str}' にPDFファイルが見つかりません。")
                st.stop()
            source_type = "folder"
            st.info(f"📁 フォルダ処理: {len(pdf_paths_to_process)}件のPDFファイル")
        else:
            st.error(f"❌ 無効なフォルダパス: '{folder_path_str}'")
            st.stop()
    elif input_mode == "📄 ファイルアップロード" and uploaded_files:
        source_type = "upload"
        st.info(f"📄 アップロード処理: {len(uploaded_files)}件のPDFファイル")
    else:
        st.sidebar.warning("⚠️ 入力ファイルを選択してください。")
        st.stop()

    # 出力先確認
    dst_dir_str = st.session_state.dst_folder_path
    if not dst_dir_str:
        st.sidebar.error("❌ 出力先フォルダが選択されていません。")
        st.stop()
        
    dst_dir = Path(dst_dir_str)
    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        st.error(f"❌ 出力先フォルダの作成失敗: {dst_dir} - {e}")
        st.stop()

    # キャッシュ設定
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        st.warning(f"⚠️ キャッシュフォルダの作成失敗: {e}")
    
    st.success(f"✅ 出力先: {dst_dir}")
    st.success(f"🚀 OCRデバイス: {device.upper()}")

    # 処理開始
    progress_bar_area = st.empty()

    if source_type == "upload":
        # アップロードファイルの一時保存
        with tempfile.TemporaryDirectory(prefix="pdf2md_gui_upload_") as upload_tmpdir_str:
            upload_tmpdir_path = Path(upload_tmpdir_str)
            temp_pdf_paths_from_upload = []
            
            for uploaded_file_data in uploaded_files:
                try:
                    temp_pdf_path = upload_tmpdir_path / uploaded_file_data.name
                    with open(temp_pdf_path, "wb") as f:
                        f.write(uploaded_file_data.getbuffer())
                    temp_pdf_paths_from_upload.append(temp_pdf_path)
                except Exception as e:
                    st.error(f"❌ ファイル保存失敗: {uploaded_file_data.name} - {e}")
            
            if not temp_pdf_paths_from_upload:
                st.error("❌ 処理対象のPDFファイルがありません。")
                st.stop()
            
            pdf_paths_to_process = temp_pdf_paths_from_upload

            # 並列処理実行
            if pdf_paths_to_process:
                total_files = len(pdf_paths_to_process)
                progress_bar = progress_bar_area.progress(0, text=f"🚀 準備中... (0/{total_files})")
                
                # プログレスコールバック
                def progress_callback(progress: float, message: str):
                    progress_bar.progress(progress, text=message)
                
                # 超高速並列処理実行
                start_time = time.time()
                results = process_pdfs_ultra_fast(pdf_paths_to_process, dst_dir, cache_dir, device, progress_callback)
                processing_time = time.time() - start_time
                
                progress_bar_area.empty()
                st.balloons()
                
                # 結果表示
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("✅ 成功", results["success"])
                with col2:
                    st.metric("⚡ キャッシュ", results["cached"])
                with col3:
                    st.metric("❌ 失敗", results["failed"])
                with col4:
                    st.metric("⏱️ 処理時間", f"{processing_time:.1f}秒")
                
                st.success(f"🎉 すべての変換が完了しました！ ({total_files}件処理)")
    else:
        # フォルダモードの場合も同様の処理
        if pdf_paths_to_process:
            total_files = len(pdf_paths_to_process)
            progress_bar = progress_bar_area.progress(0, text=f"🚀 準備中... (0/{total_files})")
            
            def progress_callback(progress: float, message: str):
                progress_bar.progress(progress, text=message)
            
            start_time = time.time()
            results = process_pdfs_ultra_fast(pdf_paths_to_process, dst_dir, cache_dir, device, progress_callback)
            processing_time = time.time() - start_time
            
            progress_bar_area.empty()
            st.balloons()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("✅ 成功", results["success"])
            with col2:
                st.metric("⚡ キャッシュ", results["cached"])
            with col3:
                st.metric("❌ 失敗", results["failed"])
            with col4:
                st.metric("⏱️ 処理時間", f"{processing_time:.1f}秒")
            
            st.success(f"🎉 すべての変換が完了しました！ ({total_files}件処理)")

# ヘルプセクション
with st.expander("📖 使い方ガイド"):
    st.markdown("""
    ### 🚀 超高速PDF→Markdown変換ツール
    
    **特徴:**
    - ⚡ CPU並列処理による超高速変換
    - 🧠 インテリジェントOCR（テキスト層がない場合のみ実行）
    - 💾 ハッシュベースキャッシュシステム
    - 📊 リアルタイム進捗表示
    
    **使い方:**
    1. **入力方式選択**: フォルダ一括処理 or 個別ファイルアップロード
    2. **出力先設定**: 変換されたMarkdownファイルの保存先
    3. **デバイス選択**: CPU or CUDA（GPU加速）
    4. **変換開始**: ワンクリックで一括変換
    
    **対応形式:**
    - 入力: PDF（テキスト付き・スキャン両対応）
    - 出力: Markdown（.md）
    """)

with st.expander("⚙️ 技術仕様"):
    st.markdown(f"""
    ### システム情報
    - **CPU コア数**: {cpu_count}
    - **並列処理**: ThreadPoolExecutor
    - **OCR エンジン**: YomiToku
    - **PDF処理**: PyMuPDF (fitz)
    - **キャッシュ**: SHA256ハッシュベース
    
    ### パフォーマンス最適化
    - DPI設定: 150（高速化）
    - バッチサイズ: CPU=4, GPU=8
    - メモリ効率: 即座にリソース解放
    - ファイルI/O: 並列書き込み
    """)
