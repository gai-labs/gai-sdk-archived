import base64
import imghdr

def read_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        data = image_file.read()
        encoded_string = base64.b64encode(data)
        base64_image =  encoded_string.decode('utf-8')
        return base64_image

def read_to_base64_imageurl(image_path):
    with open(image_path, "rb") as image_file:
        data = image_file.read()
        encoded_string = base64.b64encode(data)
        base64_image =  encoded_string.decode('utf-8')
        type = imghdr.what(None,data)
        image_url = {
            "url":f"data:image/{type};base64,{base64_image}"
        }
        return image_url

def base64_to_imageurl(base64_image):
    img_data = base64.b64decode(base64_image)
    img_type = imghdr.what(None, img_data)
    image_url = {
        "url":f"data:image/{img_type};base64,{base64_image}"
    }
    return image_url

def bytes_to_imageurl(img_data):
    img_type = imghdr.what(None, img_data)
    base64_image = base64.b64encode(img_data).decode('utf-8')
    image_url = {
        "url":f"data:image/{img_type};base64,{base64_image}"
    }
    return image_url