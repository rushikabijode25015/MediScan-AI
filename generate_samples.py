import os
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np

def create_directory_structure():
    """Ensure the static/samples folder exists"""
    os.makedirs('static/samples', exist_ok=True)
    print("Ensured static/samples directory exists.")

def generate_chest_xray(anomaly=False):
    """
    Generate a mock chest X-ray image (224x224) using PIL drawing
    """
    img = Image.new('L', (256, 256), color=15) # Dark gray background
    draw = ImageDraw.Draw(img)
    
    # 1. Draw spine (vertical stack in the middle)
    for y in range(40, 220, 12):
        draw.rectangle([123, y, 133, y+8], fill=45)
        
    # 2. Draw clavicles (shoulder bones)
    draw.line([60, 40, 120, 48], fill=65, width=4)
    draw.line([196, 40, 136, 48], fill=65, width=4)
    
    # 3. Draw lung fields (two large darker shapes)
    # Left Lung
    draw.ellipse([50, 50, 115, 200], fill=25)
    # Right Lung
    draw.ellipse([141, 50, 206, 200], fill=25)
    
    # 4. Draw rib cages (horizontal curved lines overlaying the lungs)
    for y in range(60, 190, 15):
        # Left ribs
        draw.arc([35, y, 120, y+40], start=200, end=340, fill=55, width=2)
        # Right ribs
        draw.arc([136, y, 221, y+40], start=200, end=340, fill=55, width=2)
        
    # 5. Draw heart silhouette (middle-left overlap)
    draw.ellipse([100, 110, 145, 175], fill=40)
    
    # 6. Add anomaly (simulating opacity/consolidation in pneumonia or tumor)
    if anomaly:
        # Create a blurred white cloud in the right lung (anatomical right, image left)
        # We create a separate overlay to apply a high blur
        overlay = Image.new('L', (256, 256), color=0)
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Consolidation cloud
        overlay_draw.ellipse([70, 90, 105, 135], fill=180)
        overlay_draw.ellipse([60, 110, 85, 150], fill=150)
        
        # Apply Gaussian Blur to the anomaly to make it look like a fluffy infiltrate
        overlay_blurred = overlay.filter(ImageFilter.GaussianBlur(radius=8))
        
        # Blend the anomaly with the chest X-ray
        img = Image.blend(img, Image.merge("L", (overlay_blurred,)*1), 0.5)
        
    # 7. Add medical overlays (text, scale, L/R markers)
    img_rgb = img.convert('RGB')
    draw_rgb = ImageDraw.Draw(img_rgb)
    
    # Text overlays
    draw_rgb.text((10, 10), "MediScan AI", fill=(100, 100, 100))
    draw_rgb.text((10, 25), "CXR POSTERIOR-ANTERIOR", fill=(80, 80, 80))
    draw_rgb.text((230, 10), "R", fill=(120, 120, 120)) # Patient right side is on the left, but let's just place R on image right as a label
    
    # Apply a global blur to make it look more realistic/radiographic
    img_rgb = img_rgb.filter(ImageFilter.GaussianBlur(radius=1))
    
    return img_rgb.resize((224, 224))

def generate_brain_mri(anomaly=False):
    """
    Generate a mock brain MRI image (224x224) using PIL drawing
    """
    img = Image.new('L', (256, 256), color=5) # Almost pitch black background
    draw = ImageDraw.Draw(img)
    
    # 1. Draw Skull outline
    draw.ellipse([30, 25, 226, 230], fill=20, outline=60, width=5)
    
    # 2. Draw CSF space (dark boundary inside skull)
    draw.ellipse([36, 31, 220, 224], fill=10)
    
    # 3. Draw Brain Hemispheres
    # Left hemisphere
    draw.ellipse([45, 38, 126, 217], fill=40)
    # Right hemisphere
    draw.ellipse([129, 38, 211, 217], fill=40)
    
    # 4. Draw ventricular system (central darker butterfly shape)
    # Left ventricle
    draw.ellipse([100, 105, 125, 145], fill=12)
    # Right ventricle
    draw.ellipse([130, 105, 155, 145], fill=12)
    
    # 5. Draw brain sulci/gyri (curved patterns inside hemispheres)
    for r in range(50, 120, 15):
        draw.arc([48, 40, 207, 215], start=0, end=360, fill=48, width=1)
        
    # 6. Add anomaly (simulating a bright, high-signal brain tumor)
    if anomaly:
        overlay = Image.new('L', (256, 256), color=0)
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Tumor core (bright)
        overlay_draw.ellipse([155, 85, 185, 115], fill=240)
        # Surrounding edema (darker halo)
        overlay_draw.ellipse([145, 75, 195, 125], fill=120)
        
        # Apply Gaussian Blur to the tumor components
        overlay_blurred = overlay.filter(ImageFilter.GaussianBlur(radius=5))
        
        # Blend the anomaly with the brain MRI
        img = Image.blend(img, Image.merge("L", (overlay_blurred,)*1), 0.6)
        
    # 7. Add medical overlays (text, scale, L/R markers)
    img_rgb = img.convert('RGB')
    draw_rgb = ImageDraw.Draw(img_rgb)
    
    # Text overlays
    draw_rgb.text((10, 10), "MediScan AI", fill=(100, 100, 100))
    draw_rgb.text((10, 25), "MRI BRAIN T2-WEIGHTED", fill=(80, 80, 80))
    draw_rgb.text((230, 10), "L", fill=(120, 120, 120))
    
    # Apply a global blur to make it look organic
    img_rgb = img_rgb.filter(ImageFilter.GaussianBlur(radius=1.2))
    
    return img_rgb.resize((224, 224))

def save_all_samples():
    create_directory_structure()
    
    # Chest X-Rays
    cxr_norm = generate_chest_xray(anomaly=False)
    cxr_norm.save('static/samples/cxr_normal.png')
    print("Saved cxr_normal.png")
    
    cxr_anom = generate_chest_xray(anomaly=True)
    cxr_anom.save('static/samples/cxr_anomaly.png')
    print("Saved cxr_anomaly.png")
    
    # Brain MRIs
    mri_norm = generate_brain_mri(anomaly=False)
    mri_norm.save('static/samples/mri_normal.png')
    print("Saved mri_normal.png")
    
    mri_anom = generate_brain_mri(anomaly=True)
    mri_anom.save('static/samples/mri_anomaly.png')
    print("Saved mri_anomaly.png")

if __name__ == '__main__':
    save_all_samples()
