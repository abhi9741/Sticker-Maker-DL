import streamlit
import rembg 
from PIL import Image
from io import BytesIO
import base64


streamlit.set_page_config(layout="wide")
streamlit.title("Backround Eraser / Sticker Maker")
# streamlit.write("check")

# streamlit.expander("Save this for later. Download full quality image by clicking on download result button. The webapp depends on rembg python library.",expanded=False)

with streamlit.expander("About the project"):
    streamlit.write("Stickers have become an integral aspect of social media, providing users with the opportunity to personalize their posts and express themselves in a more imaginative manner. However, the process of creating stickers can be cumbersome for regular social media users, as it requires precise manual selection of the image's subject and removal of its background. As a result, users often face restrictions in using the stickers of their choice. To overcome this challenge, we have developed a user-friendly web application that utilizes cutting-edge deep learning techniques to segment the subject of an image and erase the background, making the process simpler and more accessible. Download full quality image by clicking on download result button." )

def image_to_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

def erase_background(upload):
    image = Image.open(upload)
    col1.write("original Image")
    col1.image(image)

    image_bg_erased = rembg.remove(image)
    col2.write("Background Erased")
    col2.image(image_bg_erased)

    streamlit.sidebar.markdown("\n")
    streamlit.sidebar.download_button("Download Result",image_to_bytes(image_bg_erased), "Result.png", "image/png")

col1, col2 = streamlit.columns(2)
upload_im = streamlit.sidebar.file_uploader("Upload an image", type=["png","jpg","jpeg"])

if upload_im is not None:
    erase_background(upload_im)
else :
    erase_background("./modi.jpg")


    





