# first install opencv-python-headless numpy tensorflow
import cv2
import numpy as np
import tensorflow as tf

def load_model():
    model = tf.keras.applications.DenseNet121(weights="imagenet", include_top=False)
    return model

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    resized_image= cv2.resize(image,(512,512))
    return image, resized_image

def segment_image(model, resized_image):
    input_image = np.expand_dims(resized_image/255.0, axis=0)
    segmentation_mask= model.predict(input_image)
    segmentation_mask=np.argmax(segmentation_mask, axis=-1)[0]
    return segmentation_mask

def remove_background(image, segmentation_mask):
    mask_resized = cv2.resize(segmentation_mask, (image.shape[1],image.shape[0]), interpolation =cv2.INTER_NEAREST)
    mask_resized = mask_resized.astype(np.uint8)
    result = cv2.bitwise_and(image,image, mask=mask_resized)
    return result
def save_image(output_path,image):
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
def main(input_image_path, output_image_path):
    model = load_model()
    original_image, resized_image=preprocess_image(input_image_path)
    segmentation_mask= segment_image(model,resized_image)
    result_image= remove_background(original_image, segmentation_mask)
    save_image(output_image_path,result_image)
    print(f"Background removed and saved to {output_image_path}")
    
if __name__== "__main__":
    input_image_path = r"C:\Users\ahmad\Desktop\New folder (6)\nameLogo.png"
    output_immage_path=r"C:\Users\ahmad\Desktop\New folder (6)\outputlogo.png"
    main(input_image_path, output_immage_path)