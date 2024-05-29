import os, zipfile
import uuid
import tempfile,shutil,re
from . import constants,utils
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import TextIOWrapper
import hashlib,base64

# Remove most non-alphanumeric characters from a filename
def clean_paths(file_path_or_paths):
    if (isinstance(file_path_or_paths,list)):
        paths = []
        for file_path in file_path_or_paths:
            paths.append(clean_paths(file_path))
        return paths
    return file_path_or_paths.replace("/","_").replace("\\","_").replace(" ","_").replace(":","").replace(",","").replace("'","").replace('"','').lower()

def flatten_abs_paths(dir_or_file):
    abs_file_paths = []
    if os.path.isfile(dir_or_file):
        abs_file_paths.append(os.path.abspath(dir_or_file))
        return abs_file_paths
    
    for dirpath, _, filenames in os.walk(dir_or_file):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            abs_file_paths.append(os.path.abspath(file_path))
    return abs_file_paths

def unzip_and_remove(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_path)  # remove the .zip file after extraction

def _unzip_temp(temp_dir):
    for root, dirs, files in os.walk(temp_dir):
        for filename in files:
            if filename.endswith(".zip"):
                zip_file_path = os.path.join(root, filename)
                unzip_and_remove(zip_path=zip_file_path, extract_to=root)
                _unzip_temp(root) # recursive call to handle nested zip files

def unzip_all(file_or_dir, dest_dir=None):
    # Copy all into a temp dir
    temp_dir = tempfile.mkdtemp()
    shutil.copytree(file_or_dir, temp_dir, dirs_exist_ok=True)

    # Recursively unzip zipped files
    _unzip_temp(temp_dir)

    # Move all files to dest_dir (if exists)
    if dest_dir:
        shutil.copytree(temp_dir, dest_dir, dirs_exist_ok=True)
        shutil.rmtree(temp_dir)
        return dest_dir
    
    return temp_dir

'''
Name: get_chunk_dir
Description:
    A helper method to transform a file path or url into a subdirectory name for storing the chunks.
    It replaces the non-alphanumeric characters found as part of a path name with underscores and removes the leading underscores.
    If path_or_url is not provided, a non-hyphenated GUID named subdirectory is created instead.
Parameters:
    path_or_url: (default:None) The input text to be split into chunks.
Example:
    get_chunk_dir("/tmp/test.txt") will return "test"
    get_chunk_dir("https://www.example.com/test.txt") will return "https_www.example.com_test"
'''
def get_chunk_dir(path_or_url=None):
    if path_or_url:
        if (utils.is_url(path_or_url)):
            path_or_url = path_or_url.replace("://","_").replace("/","_").replace(".","_")
        else:
            path_or_url = os.path.basename(path_or_url).split('.')[0]
    else:
        path_or_url = str(uuid.uuid4()).replace("-","")
    return re.sub(r'^_+', '', path_or_url)

'''
Name: split_file
Description:
    The function uses LangChain's RecursiveCharacterTextSplitter to split the text into chunks.
Parameters:
    src_file: The source file path or file object
    dest_dir: (default:/tmp/chunks) The base directory for saving the chunks. The chunks will be saved in a subdirectory named after the input filename.
    chunk_size: (default:1000) The size of each chunk in characters.
    chunk_overlap: (default:100) The overlap between chunks in characters.
Output:
    dest_dir: Directory path is returned as output.    
Example:
    split_file("/tmp/test.txt",chunk_overlap=200,chunk_overlap=0) will create file at /tmp/chunks/test/chunk_id
'''
def split_file(src_file, chunk_size=1000,chunk_overlap=100):

    # Check if src_file is a file object
    if isinstance(src_file, TextIOWrapper):
        if src_file.readable():
            text = src_file.read()
            src_file = src_file.name  # Get the file path from the file object
        else:
            raise ValueError("The provided file object is not readable.")
    elif isinstance(src_file, str):
        # Check if src_file is a valid file path
        if not os.path.isfile(src_file):
            raise ValueError(f"The src_file '{src_file}' is not a valid file path.")
        # Read the text from src_file
        with open(src_file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        raise TypeError("src_file must be either a file path (str) or a file object.")

    sub_dir = get_chunk_dir(src_file)
    return split_text(text, sub_dir, chunk_size, chunk_overlap)

'''
Name: split_text
Description:
    The function uses LangChain's RecursiveCharacterTextSplitter to split the text into chunks and write each chunk into a subdirectory.
Parameters:
    text: The input text to be split into chunks.
    sub_dir: (default:None) /tmp/chunks/{sub_dir} for saving the chunks. If sub_dir is not provided, a GUID named subdirectory will be used.
    chunk_size: (default:2000) The size of each chunk in characters.
    chunk_overlap: (default:200) The overlap between chunks in characters.
Output:
    dest_dir: Directory path is returned as output.
Example:
    split_text("This is a test",chunk_overlap=200,chunk_overlap=0) will create file at /tmp/chunks/<guid>/chunk_id
    split_text("This is another test",sub_dir="test",chunk_overlap=200,chunk_overlap=0) will create file at /tmp/chunks/test/chunk_id
'''
def split_text(text, sub_dir=None ,chunk_size=2000,chunk_overlap=200):

    # Anchor to this to avoid tampering other files
    base_dir='/tmp/chunks'

    # Delete and create dest_dir
    if (sub_dir is None):
        sub_dir = str(uuid.uuid4()).replace("-","")
    dest_dir = os.path.join(base_dir,sub_dir)
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)    
    os.makedirs(dest_dir)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,        # approx 512 tokens
        chunk_overlap=chunk_overlap,     # 10% overlap
        length_function=len,
        is_separator_regex=False
    )
    chunks = splitter.create_documents([text])
    for chunk in chunks:

        # Use hash of chunk as id
        chunk_id = create_chunk_id_base64(chunk.page_content)
        chunk.metadata = {
            "chunk_id": chunk_id,
            "chunk_size": len(chunk.page_content)
        }
        chunk_fname = os.path.join(dest_dir,chunk_id)

        # Start writing the chunk
        with open(chunk_fname,'w') as f:
            f.write(chunk.page_content)
    return dest_dir

'''
Name: create_chunk_id
Description:
    The function uses sha256 to create a unique id for the chunk.
Parameters:
    text: The input text to be converted in SHA256 hash.
Example:
    create_chunk_id("This is a test") will return the SHA256 hash of the input text.
'''
def create_chunk_id(text):
    import hashlib
    if isinstance(text, bytes):
        byte_text = text
    else:
        # If 'text' is not a byte string (assuming it's a str), encode it
        byte_text = text.encode('utf-8')    
    return hashlib.sha256(byte_text).hexdigest()

'''
Generates a URL and filesystem safe Base64 encoded SHA-256 hash of the input text.

Args:
text (str or bytes): The input text to hash.

Returns:
str: The URL and filesystem safe Base64 encoded SHA-256 hash of the input text.
'''
def create_chunk_id_base64(text):
    if isinstance(text, bytes):
        byte_text = text
    else:
        # If 'text' is not a byte string (assuming it's a str), encode it
        byte_text = text.encode('utf-8')
    
    # Generate SHA256 hash (in binary form)
    hash_digest = hashlib.sha256(byte_text).digest()
    
    # Convert the binary hash to URL and filesystem safe Base64
    base64_encoded_safe = base64.urlsafe_b64encode(hash_digest).decode().rstrip('=')
    
    return base64_encoded_safe

