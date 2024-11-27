def save_mask(mask, file_path):
    # Ensure mask is 2D
    if len(mask.shape) > 2:
        mask = np.squeeze(mask, axis=-1)

    # Convert to uint8
    mask_to_save = (mask * 255).astype(np.uint8)

    # Save using PIL
    mask_image = Image.fromarray(mask_to_save)
    mask_image.save(file_path)
    print(f"Mask saved successfully to {file_path}")