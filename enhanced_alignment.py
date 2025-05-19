import os
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
import datetime

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def create_enhanced_tryon(person_img_path, cloth_img_path, output_path):
    """Create an enhanced virtual try-on result with better blending"""
    try:
        # Open images
        person_img = Image.open(person_img_path).convert('RGBA')
        cloth_img = Image.open(cloth_img_path).convert('RGBA')
        
        # Get dimensions
        person_width, person_height = person_img.size
        cloth_width, cloth_height = cloth_img.size
        
        # Resize cloth to better fit the person
        # For a shirt, we want to cover approximately the upper 40% of the body
        # This ratio works well for the sample image but might need adjustment for other images
        ratio = min(person_width * 0.75 / cloth_width, person_height * 0.38 / cloth_height)
        new_cloth_width = int(cloth_width * ratio)
        new_cloth_height = int(cloth_height * ratio)
        
        # Use high quality resizing
        cloth_img = cloth_img.resize((new_cloth_width, new_cloth_height), Image.Resampling.LANCZOS)
        
        # For this specific image, calculate position
        # We know the person's shoulders are around 25% from the top
        paste_x = (person_width - new_cloth_width) // 2
        paste_y = int(person_height * 0.22)  # Custom position for this image
        
        # Create a mask for better blending
        # We'll use the alpha channel of the garment
        r, g, b, a = cloth_img.split()
        
        # Enhance the cloth image to match lighting of the person
        enhancer = ImageEnhance.Brightness(cloth_img)
        cloth_img = enhancer.enhance(0.95)  # Slightly darken to match lighting
        
        # Apply a subtle shadow at the bottom of the garment
        shadow_overlay = Image.new('RGBA', cloth_img.size, (0, 0, 0, 0))
        shadow_gradient = np.zeros((new_cloth_height, new_cloth_width, 4), dtype=np.uint8)
        
        # Create gradient shadow (stronger at bottom)
        for y in range(new_cloth_height):
            alpha = int(min(30, max(0, (y - new_cloth_height * 0.6) / (new_cloth_height * 0.4) * 30)))
            shadow_gradient[y, :, 3] = alpha
        
        shadow_overlay = Image.fromarray(shadow_gradient)
        cloth_img = Image.alpha_composite(cloth_img, shadow_overlay)
        
        # Create a copy of the person image to work with
        result_img = person_img.copy()
        
        # Apply subtle blur to the edges of the garment for better blending
        a_blur = a.filter(ImageFilter.GaussianBlur(radius=0.5))
        cloth_img_blurred_edges = Image.merge('RGBA', (r, g, b, a_blur))
        
        # Paste the garment onto the person
        result_img.paste(cloth_img_blurred_edges, (paste_x, paste_y), cloth_img_blurred_edges)
        
        # Convert to RGB for saving (if needed)
        result_img_rgb = result_img.convert('RGB')
        
        # Save the result
        result_img.save(output_path)
        print(f"Enhanced virtual try-on created and saved to {output_path}")
        
        return result_img
    except Exception as e:
        print(f"Error creating enhanced try-on: {e}")
        return None

def create_final_image(person_img_path, cloth_img_path, result_img_path, output_path):
    """Create a final presentation image with before and after"""
    try:
        # Load images
        person_img = Image.open(person_img_path).convert('RGB')
        cloth_img = Image.open(cloth_img_path).convert('RGB')
        result_img = Image.open(result_img_path).convert('RGB')
        
        # Create a white background image
        margin = 50
        img_width = max(person_img.width, cloth_img.width, result_img.width)
        
        # Layout: Person + Cloth in first row, Result in second row
        total_width = person_img.width + cloth_img.width + 3 * margin
        total_height = max(person_img.height, cloth_img.height) + result_img.height + 4 * margin
        
        # Create a white background
        final_img = Image.new('RGB', (total_width, total_height), (255, 255, 255))
        
        # Paste person image
        x_offset = margin
        y_offset = margin
        final_img.paste(person_img, (x_offset, y_offset))
        
        # Add "+" text
        plt.figure(figsize=(1, 1), dpi=100)
        plt.text(0.5, 0.5, "+", fontsize=32, ha='center', va='center')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig("temp_plus.png", transparent=True)
        plt.close()
        
        plus_img = Image.open("temp_plus.png")
        plus_x = x_offset + person_img.width + margin//2
        plus_y = y_offset + person_img.height//2 - plus_img.height//2
        final_img.paste(plus_img, (plus_x, plus_y), plus_img.convert('RGBA'))
        
        # Paste cloth image
        x_offset = x_offset + person_img.width + margin
        final_img.paste(cloth_img, (x_offset, y_offset))
        
        # Add "=" text
        plt.figure(figsize=(1, 1), dpi=100)
        plt.text(0.5, 0.5, "=", fontsize=32, ha='center', va='center')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig("temp_equals.png", transparent=True)
        plt.close()
        
        equals_img = Image.open("temp_equals.png")
        equals_x = total_width//2 - equals_img.width//2
        equals_y = y_offset + max(person_img.height, cloth_img.height) + margin//2
        final_img.paste(equals_img, (equals_x, equals_y), equals_img.convert('RGBA'))
        
        # Paste result image centered
        x_offset = (total_width - result_img.width) // 2
        y_offset = y_offset + max(person_img.height, cloth_img.height) + margin
        final_img.paste(result_img, (x_offset, y_offset))
        
        # Save final image
        final_img.save(output_path)
        
        # Clean up temp files
        try:
            os.remove("temp_plus.png")
            os.remove("temp_equals.png")
        except:
            pass
            
        print(f"Final presentation image saved to {output_path}")
        return final_img
    except Exception as e:
        print(f"Error creating final image: {e}")
        return None

def main():
    # Input paths
    person_img_path = "zz.png"
    cloth_img_path = "shirt.png"
    
    # Ensure output directory
    ensure_dir("results")
    
    # Generate timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Create enhanced try-on
    result_path = f"results/enhanced_tryon_{timestamp}.png"
    enhanced_result = create_enhanced_tryon(person_img_path, cloth_img_path, result_path)
    
    if enhanced_result:
        # Create final presentation image
        final_path = f"results/final_presentation_{timestamp}.png"
        create_final_image(person_img_path, cloth_img_path, result_path, final_path)
        
        print("\nProcess completed successfully!")
        print(f"Enhanced try-on: {result_path}")
        print(f"Final presentation: {final_path}")
        print("\nYou can use these images for your presentation.")
    else:
        print("Failed to create enhanced try-on image.")

if __name__ == "__main__":
    main() 