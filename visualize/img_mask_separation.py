# from PIL import Image
# import os
# import numpy as np

# data_dir = os.path.join('demo_tmp', 'lady-running-65-224')
# frame_num = '18'
# image_path = os.path.join(data_dir, 'frame_00' + str(frame_num) + '.png')
# mask_path = os.path.join(data_dir, 'dynamic_mask_' + str(frame_num) + '.png')

# # Load image and mask
# image = Image.open(image_path).convert("RGBA")
# mask = Image.open(mask_path).convert("L")

# # Ensure mask is binary
# mask = mask.point(lambda x: 255 if x > 128 else 0)

# # Convert to numpy arrays
# image_np = np.array(image)
# mask_np = np.array(mask)

# # Ensure mask_np is of type uint8
# mask_np = mask_np.astype(np.uint8)

# # # Create masked and unmasked images
# # # Ensure all arrays are uint8 before stacking
# # alpha_mask = mask_np
# # inverse_alpha_mask = 255 - mask_np

# # # Stack the RGB channels with the alpha mask
# # masked_area = Image.fromarray(np.dstack((image_np[:, :, :3], alpha_mask)))
# # unmasked_area = Image.fromarray(np.dstack((image_np[:, :, :3], inverse_alpha_mask)))

# # Create masked image (transparent areas set to black)
# masked_rgb = image_np[:, :, :3].copy()
# masked_rgb[mask_np == 0] = [0, 0, 0]  # Set RGB to black where mask is 0
# masked_area = Image.fromarray(np.dstack((masked_rgb, mask_np)))

# # Create unmasked image (transparent areas set to black)
# inverse_mask_np = 255 - mask_np
# unmasked_rgb = image_np[:, :, :3].copy()
# unmasked_rgb[inverse_mask_np == 0] = [0, 0, 0]  # Set RGB to black where inverse mask is 0
# unmasked_area = Image.fromarray(np.dstack((unmasked_rgb, inverse_mask_np)))

# # Save images
# masked_area.save('frame_' + frame_num + "_masked.png")
# unmasked_area.save('frame_' + frame_num + "_unmasked.png")

from PIL import Image
import os
import numpy as np

data_dir = os.path.join('demo_tmp', 'lady-running-65-224')
frame_num = '18'
image_path = os.path.join(data_dir, 'frame_00' + str(frame_num) + '.png')
mask_path = os.path.join(data_dir, 'dynamic_mask_' + str(frame_num) + '.png')

# Load image and mask
image = Image.open(image_path).convert("RGB")  # Convert to RGB (no alpha channel)
mask = Image.open(mask_path).convert("L")      # Convert to grayscale

# Ensure mask is binary
mask = mask.point(lambda x: 255 if x > 128 else 0)

# Convert to numpy arrays
image_np = np.array(image).astype(np.uint8)
mask_np = np.array(mask).astype(np.uint8)

# Create masked image (masked area kept, unmasked area set to black)
masked_image_np = image_np.copy()
masked_image_np[mask_np == 0] = [0, 0, 0]  # Set unmasked areas to black
masked_area = Image.fromarray(masked_image_np)

# Create unmasked image (unmasked area kept, masked area set to black)
unmasked_image_np = image_np.copy()
unmasked_image_np[mask_np == 255] = [0, 0, 0]  # Set masked areas to black
unmasked_area = Image.fromarray(unmasked_image_np)

# Save images
masked_area.save('frame_' + frame_num + "_masked.png")
unmasked_area.save('frame_' + frame_num + "_unmasked.png")