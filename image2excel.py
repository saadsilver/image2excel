# 📦 Standard library
import os
import sys
import io
import re
import zipfile
import tempfile
from datetime import datetime

# 🖼️ Image and PDF processing
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from pdf2image import convert_from_bytes

# 🔤 OCR
import pytesseract

# 📊 Data and plotting
import pandas as pd
import matplotlib.pyplot as plt

# 🌐 Streamlit app
import streamlit as st


# Configuration Tesseract
if os.name == 'nt':  # Configuration Windows
    possible_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.join(os.getenv('LOCALAPPDATA', ''), r"Tesseract-OCR\tesseract.exe")
    ]
    tesseract_path = None
    for path in possible_paths:
        if os.path.exists(path):
            tesseract_path = path
            break
    
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        tessdata_dir = os.path.dirname(tesseract_path)
        os.environ["TESSDATA_PREFIX"] = os.path.join(tessdata_dir, "tessdata")
    else:
        # Si non trouvé, on espère qu'il est dans le PATH, sinon afficher un warning
        print("⚠️ Tesseract non trouvé dans les dossiers standards. Assurez-vous qu'il est installé et dans le PATH.")
else:
    # Linux / Streamlit Cloud : Tesseract est généralement dans le PATH
    pass

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Extracteur de Tableaux de Factures",
    page_icon="📊",
    layout="wide"
)


def extract_lines(binary_image, line_scale=40):
    """
    Extract horizontal and vertical lines from a binary image using dynamic kernel sizes.
    
    Args:
    - binary_image: Grayscale binary image.
    - line_scale: Divider to determine kernel size based on image dimensions (default 40).
                  Larger value = smaller kernel.
    """
    h, w = binary_image.shape
    
    # Dynamic kernel size based on image width
    # Prevents hardcoded values from failing on high-res/low-res images
    line_len = max(10, w // line_scale) 
    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
    horizontal_mask = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, horizontal_kernel)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_len))
    vertical_mask = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, vertical_kernel)

    lines_mask = cv2.bitwise_or(horizontal_mask, vertical_mask)
    return cv2.bitwise_not(lines_mask)

def minimize_contour(contour, n_points=4):
    # Approximate the contour to 'n_points' points
    epsilon = 0.02 * cv2.arcLength(contour, True)  # Approximation accuracy factor
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Reduce the number of points to exactly n if needed
    while len(approx) > n_points:
        epsilon *= 1.1  # Increase epsilon until the number of points is <= n
        approx = cv2.approxPolyDP(contour, epsilon, True)

    return approx


def create_mask(image, contour):
    """
    Creates a mask from a contour and applies it to the original image.
    
    Args:
        image: The original image (BGR).
        contour: The contour to create the mask from.
    
    Returns:
        masked_image: The original image with the mask applied.
    """
    # Step 1: Create a blank black mask (same size as the image)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Step 2: Draw the contour on the mask (white)
    cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

    # Step 3: Apply the mask to the original image using bitwise AND
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    return masked_image


def extract_grid(masked_image):
    grey_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grey_image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    if binary is None or binary.size == 0:
        raise ValueError("Erreur : L'image seuillée est vide.")
    
    h, w = binary.shape
    line_len = max(20, w // 50)  # Dynamic size
    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_len))
    
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    
    # Dilate slightly to ensure intersection
    horizontal_lines = cv2.dilate(horizontal_lines, np.ones((2, 2), np.uint8), iterations=2)
    vertical_lines = cv2.dilate(vertical_lines, np.ones((2, 2), np.uint8), iterations=2)
    
    table_structure = 255 - cv2.addWeighted(horizontal_lines, 1, vertical_lines, 1, 0)
    
    return table_structure, horizontal_lines, vertical_lines  # Return separated lines for better cell detection

def sort_contours_topleft(contours, row_threshold=15):
    """
    Sort contours by row (using y-center) then by column (x-coordinate).
    Standard cv2.boundingRect(c) is (x, y, w, h).
    """
    # Calculate bounding boxes and centers
    boxes = [cv2.boundingRect(c) for c in contours]
    
    # Store (contour, x, y, w, h, center_y)
    contours_with_info = []
    for c, box in zip(contours, boxes):
        x, y, w, h = box
        center_y = y + h // 2
        contours_with_info.append((c, x, y, w, h, center_y))
    
    # Sort initially by Y-center
    contours_with_info.sort(key=lambda b: b[5])

    sorted_contours = []
    current_row = []
    
    for item in contours_with_info:
        if not current_row:
            current_row.append(item)
        else:
            prev_center_y = current_row[-1][5]
            curr_center_y = item[5]
            
            # Check if in same row (centers are close)
            if abs(curr_center_y - prev_center_y) <= row_threshold:
                current_row.append(item)
            else:
                # Process finished row: Sort by X
                current_row.sort(key=lambda b: b[1])
                sorted_contours.extend([x[0] for x in current_row])
                current_row = [item]
    
    # Add last row
    if current_row:
        current_row.sort(key=lambda b: b[1])
        sorted_contours.extend([x[0] for x in current_row])

    return sorted_contours

def find_bounding_box(binary_image):
    # Get indices of foreground pixels (value == 1)
    rows, cols = np.where(binary_image == 255)

    # If there are no foreground pixels, return None
    if rows.size == 0 or cols.size == 0:
        return None

    # Determine bounding box coordinates
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()

    # Top-left and bottom-right coordinates
    top_left = (min_row, min_col)
    bottom_right = (max_row, max_col)

    return top_left, bottom_right

def extract_dimensions(table_structure):
    top_left, bottom_right = find_bounding_box(table_structure)
    if top_left is None or bottom_right is None:
        # Handle empty or invalid bounding box
        return 0, 0

    row_ = np.max(table_structure[top_left[0]:top_left[0]+100, :], axis=0)
    col_ = np.max(table_structure[:, top_left[1]:top_left[1]+100], axis=1)
    n_col = np.sum(np.diff(row_) > 200)
    n_row = np.sum(np.diff(col_) > 200)
    return n_row, n_col

def clean_extracted_data(table_data, n_row, n_col):
    try:
        # Reformatage en DataFrame
        data = [table_data[i:i + n_col] for i in range(0, len(table_data), n_col)]
        df = pd.DataFrame(data)
        df.columns = [f"Colonne_{i+1}" for i in range(n_col)]

        # 🔹 Nettoyage brut initial
        for col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].str.replace("VIDE", "", regex=False)
            df[col] = df[col].str.replace(" ", "", regex=False)
            df[col] = df[col].str.replace("O", "0")
            df[col] = df[col].str.replace("o", "0")
            df[col] = df[col].str.replace("I", "1")
            df[col] = df[col].str.replace("l", "1")

        # 🔹 Suppression des lignes parasites (en-têtes, lignes vides)
        df = df[~df.apply(lambda row: row.astype(str).str.contains(r'N°|Colonne|Page|Montant', case=False).any(), axis=1)]
        df = df.dropna(how="all").reset_index(drop=True)

        # 🔹 Si structure à 6 colonnes connue
        if n_col == 6:
            df.columns = ["ID", "N° Facture", "Nombre de pages", "N° Client", "N° Abonnement", "Montant HT"]

        # 🔹 Nettoyage spécifique - N° Abonnement
        if "N° Abonnement" in df.columns:
            def clean_abonnement(val):
                val = str(val).strip()
                val = val.replace(" ", "").replace("O", "0").replace("o", "0")

                if "facture" in val.lower():
                    return "Facture groupée"

                # Pour les numéros, on ne garde que les alphanumeric majuscules et chiffres
                clean_val = re.sub(r"[^A-Z0-9]", "", val.upper())
                
                # Supprimer les bruits courts (souvent des coches interprétées comme 'I' ou '1')
                if len(clean_val) < 5: 
                    return ""
                return clean_val

            df["N° Abonnement"] = df["N° Abonnement"].apply(clean_abonnement)

        # 🔹 Nombre de pages → tout à 1
        if "Nombre de pages" in df.columns:
            df["Nombre de pages"] = 1

        # 🔹 Montant HT → conversion robuste
        if "Montant HT" in df.columns:
            def convert_to_clean_float(val):
                try:
                    # Enlever tout sauf chiffres, points, virgules
                    val = re.sub(r"[^0-9,.]", "", str(val))
                    val = val.replace(',', '.')
                    # Trouver le dernier nombre qui ressemble à un montant
                    match = re.search(r'\d+\.\d{2}', val)
                    if not match:
                         match = re.search(r'\d+\.\d+', val)
                    return abs(float(match.group())) if match else None
                except:
                    return None
            df["Montant HT"] = df["Montant HT"].apply(convert_to_clean_float)
            df = df[df["Montant HT"].notna()]  # supprime les lignes invalides
            df["Montant HT"] = df["Montant HT"].apply(
                lambda x: "{:.2f}".format(x) if isinstance(x, (int, float)) and pd.notna(x) else x
            )

        # 🔹 Nettoyage N° Facture (garder que chiffres)
        if "N° Facture" in df.columns:
            df["N° Facture"] = df["N° Facture"].astype(str).str.replace(r"\D", "", regex=True)

        # 🔹 Nettoyage ID
        if "ID" in df.columns:
            df["ID"] = df["ID"].str.extract(r'(\d+)', expand=False)
            df["ID"] = pd.to_numeric(df["ID"], errors="coerce")
            df = df[df["ID"].notnull()].astype({"ID": int})

        # 🔹 Nettoyage N° Client (format partiel)
        if "N° Client" in df.columns:
            def clean_client(client):
                client = str(client)
                # Keep mostly digits and dots, remove letters unless it looks like a real ID
                # Filter out garbage like "assesai0se"
                if re.search(r'[A-Za-z]{3,}', client): # If valid client ID shouldn't have many letters
                    return ""
                client = re.sub(r'[^\d.]', '', client)
                # Client id usually starts with 7.
                if len(client) < 5: return ""
                return client
            df["N° Client"] = df["N° Client"].apply(clean_client)

        return df.reset_index(drop=True)

    except Exception as e:
        print(f"❌ Erreur lors du nettoyage : {e}")
        return pd.DataFrame()


def process_image(image):
    """
    Process image and extract table data using optimized batch OCR.
    """
    try:
        # Étape 1: Conversion en binaire
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Étape 2: Extraction des lignes (avec échelle dynamique)
        out = extract_lines(binary_image, line_scale=40)
        
        # Étape 3: Trouver les contours
        contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, "Aucun contour détecté dans l'image"
        
        # Étape 4: Trouver le plus grand contour (cadre du tableau)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Étape 5: Créer le masque
        masked_image = create_mask(image, minimize_contour(largest_contour))
        
        # Étape 6: Extraire la structure de la grille
        table_structure, h_lines, v_lines = extract_grid(masked_image)
        
        # Étape 7: Trouver les cellules
        contours, _ = cv2.findContours(table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sort_contours_topleft(contours)
        
        # Étape 8: Extraire les dimensions
        n_row, n_col = extract_dimensions(table_structure)
        
        # Étape 9: OCR en batch sur TOUTE l'image
        # Prétraitement avancé pour nettoyer l'image
        ocr_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        
        # 1. Mettre le fond (hors masque) en blanc au lieu de noir
        # Le masque est blanc (255) dans le ROI et noir (0) ailleurs
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [minimize_contour(largest_contour)], -1, (255), thickness=cv2.FILLED)
        ocr_image[mask == 0] = 255  # Force white background outside ROI
        
        # 2. Effacer les lignes de la grille (les mettre en blanc)
        # Dilater légèrement les lignes pour s'assurer qu'elles couvrent bien les traits
        kernel_clean = np.ones((3,3), np.uint8)
        lines_mask = cv2.add(h_lines, v_lines)
        lines_mask = cv2.dilate(lines_mask, kernel_clean, iterations=1)
        
        # Appliquer la "gomme" sur les lignes
        ocr_image = cv2.add(ocr_image, lines_mask) # Add white lines to image (saturates to 255)
        
        # 3. Ajouter une bordure blanche finale
        ocr_image = cv2.copyMakeBorder(ocr_image, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        
        # 4. Binarisation finale
        _, ocr_binary = cv2.threshold(ocr_image, 160, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Configuration : psm 6 pour un bloc de texte uniforme
        config = "--psm 6 -l fra+eng --oem 3"
        
        # Récupère les données détaillées (mots avec coordonnées)
        data = pytesseract.image_to_data(ocr_binary, config=config, output_type=pytesseract.Output.DICT)
        
        # Préparation des conteneurs pour les cellules
        cell_data = [""] * len(sorted_contours)
        cell_rects = [cv2.boundingRect(c) for c in sorted_contours]
        
        n_boxes = len(data['level'])
        
        # Offset dû au border ajouté (20px)
        offset_x, offset_y = 20, 20
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            if not text:
                continue
                
            # Coordonnées du mot dans l'image OCR
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            
            # Coordonnées réelles dans l'image originale
            real_x = x - offset_x
            real_y = y - offset_y
            real_center_x = real_x + w // 2
            real_center_y = real_y + h // 2
            
            # Trouver à quelle cellule appartient ce mot
            # Optimisation: vérifier intersection simple
            best_idx = -1
            
            for idx, (cx, cy, cw, ch) in enumerate(cell_rects):
                # Vérifie si le centre du mot est dans la cellule
                if (cx < real_center_x < cx + cw) and (cy < real_center_y < cy + ch):
                    best_idx = idx
                    break
            
            if best_idx != -1:
                # Ajoute le texte à la cellule correspondante
                current_text = cell_data[best_idx]
                if current_text:
                    cell_data[best_idx] += " " + text
                else:
                    cell_data[best_idx] = text
        
        # Post-traitement du texte comme avant
        final_data = []
        for text in cell_data:
            text = text.replace('\n', ' ').replace('|', '').replace('O', '0').replace('o', '0').replace('I', '1').replace('l', '1').strip()
            final_data.append(text if text else "VIDE")
            
        # Étape 10: Nettoyer et structurer les données
        df = clean_extracted_data(final_data, n_row, n_col)
        
        return df, f"Extraction réussie! {len(sorted_contours)} cellules détectées, {n_row} lignes, {n_col} colonnes"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Erreur lors du traitement: {str(e)}"
def main():
    st.title("Extracteur de tableaux depuis PDF ou Image")

    with st.expander("🔍 Diagnostic Debug Info", expanded=False):
        st.write(f"CWD: {os.getcwd()}")
        st.write(f"Python: {sys.executable}")
        st.write(f"Tesseract CMD: {pytesseract.pytesseract.tesseract_cmd}")
        st.write(f"Exists? {os.path.exists(pytesseract.pytesseract.tesseract_cmd)}")
        try:
            import subprocess
            res = subprocess.run([pytesseract.pytesseract.tesseract_cmd, "--version"], capture_output=True, text=True)
            st.code(res.stdout)
        except Exception as e:
            st.error(f"Execution failed: {e}")

    uploaded_file = st.file_uploader(
        "Uploader un fichier (PDF ou Image)",
        type=["pdf", "png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        images = []
        
        # Vérifier le type de fichier
        if uploaded_file.type == "application/pdf":
            # Cas PDF
            pdf_bytes = uploaded_file.read()
            with st.spinner("Conversion du PDF en images..."):
                images = convert_from_bytes(pdf_bytes, dpi=400)
            st.success(f"{len(images)} pages détectées via PDF")
            
        else:
            # Cas Image
            try:
                image = Image.open(uploaded_file)
                images = [image]
                st.success("Image chargée avec succès")
            except Exception as e:
                st.error(f"Erreur lors de la lecture de l'image : {e}")
                return

        # Choix de la page
        page_number = st.selectbox(
            "Choisir la page à analyser",
            options=list(range(1, len(images) + 1))
        )

        # Image sélectionnée
        selected_image = images[page_number - 1]

        st.subheader(f"Aperçu de la page {page_number}")
        st.image(selected_image, use_column_width=True)

        # Conversion PIL → OpenCV
        img = np.array(selected_image)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Lancer l'OCR
        if st.button("Extraire le tableau"):
            with st.spinner("Extraction du tableau..."):
                df, msg = process_image(img_bgr)

            st.info(msg)

            if df is not None and not df.empty:
                st.dataframe(df, use_container_width=True)

                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    df.to_excel(writer, index=False, sheet_name="Extraction")
                excel_bytes = output.getvalue()
                
                st.download_button(
                    "Télécharger Excel",
                    data=excel_bytes,
                    file_name=f"extraction_page_{page_number}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("Aucune donnée détectée sur cette page.")


if __name__ == "__main__":
    main()